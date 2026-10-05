# Tools reference

Eleven tools run inference through `DecisionBridge.predict(state, preset)` or `DecisionBridge.predict_custom(state, questions, state_key)`. Each wraps an engine-registered preset (or a custom question set) and parses the result into a typed Pydantic output schema. Two more (`decision_update`, `decision_usage`) are maintenance tools that don't run the models.

## How the engine answers

Every `DecisionEngine` returns `{"answers": {<question>: <answer>}, "usage": ..., "routing": ...}`. Each answer is one of three typed shapes, and every tool parses them with the helpers in `decision_mcp/tools/_helpers.py`:

| Question type | Answer field | Meaning |
| ------------- | ------------ | ------- |
| `choice` | `choice` | The winning label, plus per-label `probabilities` |
| `noul` | `noul` | Probability of "yes" (tools threshold at 0.5) |
| `score` | `score` + `legend` | Expected level `0..N-1` with a text legend, plus per-level `probabilities` |

Every answer also has `answer_confidence` in `[0, 1]`, which is what the tools report as `confidence`. `details` in each tool's output is the full raw engine result (including which checkpoint answered).

The text is passed to the engine as `{<field>: text}` because each question's instructions refer to its input by name (`prompt`, `request`, `post`, `message`, `body` for the presets; see the engine's `presets()` / `PresetSpec.state_key`).

## decision_guard

Detect prompt-injection / jailbreak attempts.

**Input:**
```python
class GuardInput(BaseModel):
    prompt: str
```

**Output:**
```python
class GuardOutput(BaseModel):
    is_injection: bool
    confidence: float    # 0.0–1.0
    details: dict        # raw engine response
```

**When to use:** Before sending any untrusted text to the LLM. Run on:
- User-submitted prompts
- Email content from external senders
- Document content fetched from the web
- Tool return values containing user-controlled data

**How it decides:** `is_injection` is true when the `jailbreak` or `prompt_injection` question has P(yes) ≥ 0.5. `confidence` is the probability of the verdict returned (P(yes) if flagged, 1 − P(yes) if not).

**Cost:** ~33 ms (T4). Much cheaper than asking a frontier model to check.

**Limitations:**
- Trained on a specific distribution of injection attacks. Novel attack vectors may evade it.
- Use as **one signal among many**, not the only signal.
- Pair with input length limits, output filtering, and privilege boundaries.

**Where the parser lives:** `decision_mcp/tools/guard.py`

---

## decision_route

Decide whether a prompt needs a small/cheap model or a frontier/smart model.

**Input:**
```python
class RouteInput(BaseModel):
    prompt: str
```

**Output:**
```python
class RouteOutput(BaseModel):
    tier: str            # "small" | "frontier"
    confidence: float    # 0.0–1.0
    details: dict
```

**When to use:** At the start of every agent turn, to pick a model. Saves API spend dramatically for trivial prompts ("hi", "thanks", "yes/no questions").

**How it decides:** the engine's `difficulty` question has four levels (trivial / easy / moderate / hard). The tier is `frontier` when at least 50% of the probability mass is on moderate + hard, otherwise `small`. `confidence` is the mass on the side chosen. To change the cut-off, edit `_FRONTIER_LEVELS` in `decision_mcp/tools/route.py`.

**Cost:** ~33 ms (T4). Saves potentially hundreds of milliseconds + dollars per call when routing to a small model.

**Where the parser lives:** `decision_mcp/tools/route.py`

---

## decision_triage

Classify a support ticket: intent, urgency, churn risk, refund request and frustration.

**Input:**
```python
class TriageInput(BaseModel):
    text: str
```

**Output:**
```python
class TriageOutput(BaseModel):
    intent: str              # refund | technical_help | billing_question | information | cancellation | other
    is_urgent: bool
    churn_risk: bool
    refund_requested: bool
    frustration: str         # "calm and neutral" | "concerned but civil" | "clearly annoyed" | "very angry or using strong language"
    confidence: float        # confidence in `intent`
    details: dict
```

**When to use:** First step in any support-ticket-handling workflow. After classifying, route to the right subagent based on `intent`.

**Cost:** ~33 ms.

**Upstream questions used:** `intent` (choice), `is_urgent` / `refund_requested` / `churn_risk` (noul), `frustration` (score, four levels; the legend text is returned). These come from the engine's `triage` preset; if an engine upgrade renames them the tool raises `ToolError` naming the missing key.

**Where the parser lives:** `decision_mcp/tools/triage.py`

---

## decision_moderate

Check text for toxicity, harassment, and threats.

**Input:**
```python
class ModerateInput(BaseModel):
    text: str
```

**Output:**
```python
class ModerateOutput(BaseModel):
    is_toxic: bool
    is_harassment: bool
    is_threat: bool
    confidence: float    # the weakest of the three verdicts
    details: dict
```

`details` also carries the engine's `spam` and `severity` answers.

**When to use:** Before any user-facing output is shown. Run on:
- Agent responses that get posted to public channels
- User-submitted content in community spaces
- Inbound messages from unknown senders

**Limitations:**
- Not a substitute for human moderation.
- False positives possible (sarcasm, reclaimed slurs, in-group speech).
- False negatives possible (especially for novel attacks or coded language).

**Where the parser lives:** `decision_mcp/tools/moderate.py`

---

## decision_email

Email-specific triage: owning team, urgency, needs-reply, spam and phishing.

**Input:**
```python
class EmailInput(BaseModel):
    body: str
```

**Output:**
```python
class EmailOutput(BaseModel):
    category: str        # billing | technical | sales | security | hr | other
    urgency: str         # "no time pressure" | "needs attention soon" | "blocking issue or hard deadline"
    needs_reply: bool
    is_spam: bool
    is_phishing: bool
    confidence: float    # confidence in `category`
    details: dict
```

**When to use:** First step in any inbox-management workflow. Run on each new email to decide if it needs attention.

**How it decides:** the boolean fields are P(yes) ≥ 0.5 on the engine's `needs_reply`, `is_spam` and `is_phishing` questions.

**Cost:** ~33 ms.

**Where the parser lives:** `decision_mcp/tools/email.py`

---

## Coding-specific tools

The six tools below use **custom question schemas** (not engine presets) and are tuned for coding workflows. They all go through `DecisionBridge.predict_custom()` — `bridge.predict()` is reserved for the five engine presets above.

Each defines its questions as a plain dict at the top of its file (`{"type": "choice", "instructions": "... `text` ...", "criteria": {label: description}}`). Two things learned while tuning them against the real models:

- **Use `choice` for ordinal scales too** (low / medium / high, S0–S3, skip → critical). The `score` type gave noticeably worse answers on these custom questions (e.g. a `nit` comment rated medium priority, a README typo rated S2); the same scales as `choice` fixed most of them.
- **Custom questions are not what the engine was trained on**, so treat `confidence` on them as a rough signal. Coarse labels (tone, type, area, kind of secret) are reliable; fine-grained ones (`scope`, `risk`) are less so. `tests/test_integration.py` shows what is verified.

---

## decision_review_tone

Classify a code review comment's tone and priority.

**Input:**
```python
class ReviewToneInput(BaseModel):
    comment: str
```

**Output:**
```python
class ReviewToneOutput(BaseModel):
    tone: str          # nit | suggestion | blocking | praise | question | off_topic
    priority: str      # low | medium | high
    confidence: float  # 0.0–1.0
    details: dict
```

**When to use:**
- Before deciding whether to reply to a review comment at all (skip `off_topic`)
- Routing tone-suggestion replies to small models, blocking-tone replies to frontier
- Triaging a backlog of review comments by priority

**Limitations:** Trained on review-comment-like text. Won't work well on Slack chatter or unrelated prose.

**Where the parser lives:** `decision_mcp/tools/review_tone.py`

---

## decision_bug_severity

Classify a bug report's severity and area.

**Input:**
```python
class BugSeverityInput(BaseModel):
    text: str
```

**Output:**
```python
class BugSeverityOutput(BaseModel):
    severity: str      # S0_critical | S1_high | S2_medium | S3_low
    area: str          # frontend | backend | infra | docs | tests | deps | auth | unknown
    confidence: float  # 0.0–1.0
    details: dict
```

**When to use:**
- Auto-triage incoming bug reports (GitHub Issues, Linear, email)
- Surface S0/S1 bugs to on-call rotations
- Group bugs by area for sprint planning

**S0 vs S1:** S0 = "users are blocked right now" (auth down, data loss, security). S1 = "users are degraded" (slow, broken edge case, intermittent).

**Where the parser lives:** `decision_mcp/tools/bug_severity.py`

---

## decision_commit_classify

Classify a commit message by type, scope, and risk.

**Input:**
```python
class CommitClassifyInput(BaseModel):
    message: str
```

**Output:**
```python
class CommitClassifyOutput(BaseModel):
    type: str          # feat | fix | refactor | chore | docs | test | perf | build | ci | revert
    scope: str         # api | ui | db | infra | deps | auth | none
    risk: str          # low | medium | high
    confidence: float  # 0.0–1.0
    details: dict
```

**When to use:**
- Normalize commit messages that don't follow Conventional Commits
- Auto-tag PRs based on commit content
- Surface high-risk commits for review

**Risk signal:** "high" doesn't mean broken — it means "this changes auth / db / infra / public API surface". Use as a routing signal not a quality signal.

**Where the parser lives:** `decision_mcp/tools/commit_classify.py`

---

## decision_test_priority

Classify a test's run priority and the reason.

**Input:**
```python
class TestPriorityInput(BaseModel):
    description: str   # test name + description, or full test body
```

**Output:**
```python
class TestPriorityOutput(BaseModel):
    priority: str     # skip | low | medium | high | critical
    reason: str        # covers_new_code | covers_bug_fix | covers_regression | smoke_test | redundant | flaky
    confidence: float  # 0.0–1.0
    details: dict
```

**When to use:**
- Decide test execution order on a CI runner with limited time
- Skip `redundant` / `flaky` tests in fast-feedback loops
- Surface `critical` / `covers_regression` tests for pre-merge runs

**Where the parser lives:** `decision_mcp/tools/test_priority.py`

---

## decision_secret_risk

Detect leaked secrets / credentials in text.

**Input:**
```python
class SecretRiskInput(BaseModel):
    text: str
```

**Output:**
```python
class SecretRiskOutput(BaseModel):
    kind: str          # none | api_key | password | token | cert | ssh_key | aws_creds | other
    risk: str          # none | low | medium | high | critical
    confidence: float  # 0.0–1.0
    details: dict
```

**When to use:**
- Before saving text to memory (catch leaked creds before they're persisted)
- Before posting text to public channels
- Before including text in commit messages / PR descriptions

**Different from `decision_guard`:** `decision_guard` detects prompt-injection (attempts to manipulate the agent). `decision_secret_risk` detects leaked credentials (data exposure). They're separate concerns — run both.

**Where the parser lives:** `decision_mcp/tools/secret_risk.py`

---

## decision_diff_intent

Classify a PR diff: intent, scope, risk.

**Input:**
```python
class DiffIntentInput(BaseModel):
    diff: str   # unified diff text
```

**Output:**
```python
class DiffIntentOutput(BaseModel):
    intent: str       # add_feature | fix_bug | refactor | perf | docs | test | build | chore | revert
    scope: str        # single_file | module | cross_cutting
    risk: str         # low | medium | high
    confidence: float # 0.0–1.0
    details: dict
```

**When to use:**
- Auto-generate PR descriptions from the diff
- Route PRs by intent (e.g., `chore` → fast-track, `cross_cutting` → senior reviewer)
- Trigger different CI pipelines by intent (e.g., `perf` → run benchmarks)

**`risk` here** = "blast radius if this breaks", not "is this buggy". `cross_cutting` + `high` = needs deep review. `single_file` + `low` = likely safe.

**Where the parser lives:** `decision_mcp/tools/diff_intent.py`

---

## Maintenance tools

These two don't run the models.

---

## decision_update

Check whether the decision engine (currently the Laya library) or its model weights are out of date, and whether new Laya models exist. Laya has no update mechanism of its own.

**Input:** `action`: `"check"` (default, read-only) or `"apply"`.

**Output:**
```python
class UpdateOutput(BaseModel):
    laya_installed: str
    laya_latest: str                 # from PyPI
    laya_update_available: bool
    model_repos: list[RepoStatus]    # repo, models served, cached_commit, remote_commit, update_available
    new_models: list[str]            # Hub models this laya version doesn't load (report-only)
    applied: list[str]               # what `apply` did
    restart_required: bool
```

(The `laya_*` field names are intentional: this tool is the Laya engine's updater. When a second engine lands, it brings its own updater.)

**What `check` compares:**
- installed `laya` vs. the latest release on PyPI
- the commit of each model repo in the local cache (`./models`) vs. the Hub head
- new models: extra top-level model folders in `convaiinnovations/laya` (each has an `rl_agent_config.json`) and any other `convaiinnovations/laya*` repo that this laya version doesn't list. New models are only reported, never downloaded, because an older `laya` may not be able to load them; upgrade first.

**`apply`:** runs `pip install --upgrade laya` (via `uv pip` when available) and re-downloads model repos whose commit changed. It only ever installs the `laya` package. Because it runs pip and the HTTP server has no auth, it is **refused unless `DECISIONMCP_ALLOW_UPDATES=true`**. Nothing takes effect in the running process, so `restart_required` is set: restart `decisionmcp` afterwards, then run `check` again (a new laya may list new models).

Network failures (PyPI or the Hub unreachable) raise `ToolError` rather than returning a partial answer.

**Where it lives:** `decision_mcp/tools/update.py`

---

## decision_usage

Summarise what the engine has been used for, or export it as training data. Engines only expose per-call metadata (`run_id`, token usage, timing) and persist nothing, so DecisionMCP keeps its own log.

**Input:** `action`: `"stats"` (default) or `"export"`; optional `since_hours`; for `export`, optional `labeled_only` (only rows that carry input text — ready `(input, answers)` training pairs; the export file is then named `usage-labeled-<timestamp>.jsonl`).

**What is logged:** every prediction, success or failure, in a SQLite file (`<data_dir>/usage.db`, default `./data/usage.db`): timestamp, MCP session id, tool, preset, checkpoint that answered, latency, input length and tokens, the engine's full answers (labels + probabilities) and routing, plus the engine version that produced the labels (`engine_version` — rows written before this column existed carry `null`: provenance unknown; legacy `laya_version` columns are renamed in place). One `sessions` row per MCP connection records the client's user-agent and when it disconnected.

**Input text is NOT stored by default**, because inputs can hold prompts, source code and credentials. Set `DECISIONMCP_USAGE_STORE_TEXT=true` to keep it; that is what makes an export usable as `(input, answers)` pairs. Even then, text sent to `decision_secret_risk` is never stored.

**`export`** writes successful calls as JSONL to `<data_dir>/exports/usage-<timestamp>.jsonl` and returns the path and row count (the file path is never caller-supplied). Each line: `{run_id, ts, session_id, tool, preset, model, engine_version, input, answers}`, where `input` is `null` for calls made while text storage was off. Pass `labeled_only=true` to skip those rows entirely.

The engine answers are the model's own predictions, not ground truth. **This log is a raw capture store**: everything is written as the model produced it, with no capture-time filtering, curation, or cleanup (the only carve-out is the secret-risk exclusion above). Sanitization, dedup and label review are downstream steps applied to exports — never in the capture path.

Disable logging with `DECISIONMCP_USAGE_ENABLED=false` (the tool then returns an error).

**Where it lives:** `decision_mcp/tools/usage_report.py` (tool), `decision_mcp/usage.py` (store)

---

## Error handling

Tools **raise on any unexpected engine output** rather than returning silent defaults. The server catches these and returns them as MCP error content blocks (with `isError=true`).

### Exception hierarchy

```
DecisionMCPError        (catch this for any project-specific failure)
├── ModelLoadError       (engine weights couldn't load)
├── UnknownPresetError  (tool requested a preset that doesn't exist)
└── BridgeError         (engine call failed: GPU OOM, runtime error, etc.)
```

`ToolError` (also extends `DecisionMCPError`) is raised when a tool's parser can't extract what it needs from the engine's response. It carries the tool name and an optional `cause`.

### When a tool raises

| Failure                                        | Raised by      | Server response to client                  |
| ---------------------------------------------- | -------------- | ------------------------------------------ |
| The engine raises (GPU OOM, runtime error)     | `BridgeError`  | MCP error block: `"[decision_guard] Engine predict failed (preset='guard', state_len=12)"` |
| Tool requested a preset not in `DecisionBridge.presets` | `UnknownPresetError` | MCP error block: `"[decision_guard] Unknown preset 'bogus'. Choose from [...]"` |
| Engine returned a non-dict                     | `ToolError`    | MCP error block: `"[decision_guard] Expected dict from engine, got str: 'oops'"` |
| Engine returned an empty dict                  | `ToolError`    | MCP error block: `"[decision_guard] Engine returned an empty result."` |
| Expected question missing in engine output     | `ToolError`    | MCP error block: `"[decision_guard] Missing expected key 'jailbreak' in engine answers. Got keys: []"` |
| Answer has the wrong type for the question     | `ToolError`    | MCP error block: `"[decision_guard] Expected 'noul' answer at 'jailbreak', got type 'choice'"` |
| Confidence isn't numeric                       | `ToolError`    | MCP error block: `"[decision_guard] Confidence at 'jailbreak' is not numeric: 'NaN'"` |
| PyPI / Hugging Face Hub unreachable (`decision_update`) | `ToolError` | MCP error block: `"[decision_update] Could not query the Hugging Face Hub: ..."` |
| `decision_update` `apply` while updates are disabled | `ToolError` | MCP error block: `"[decision_update] Updates are disabled. Set DECISIONMCP_ALLOW_UPDATES=true ..."` |
| Tool input fails schema validation             | `ValidationError` (Pydantic) | MCP error block: `"Invalid input for decision_guard: ..."` |
| Unknown tool name                              | (server-side)  | MCP error block: `"Unknown tool 'foo'. Available: ['decision_guard', ...]"` |

### What the server logs

All `DecisionMCPError`s are logged with **full traceback** at `ERROR` level, including the input keys (never values — to avoid leaking prompts):

```
2024-01-15 12:34:56 ERROR decision_mcp.server Tool decision_guard failed (input_keys=['prompt']): [decision_guard] Expected dict from engine, got str: 'oops'
Traceback (most recent call last):
  ...
```

Unexpected exceptions (not `DecisionMCPError`) are also logged with traceback, but the **caller sees only** `"Internal error in decision_guard. Check server logs."` to prevent leaking internals.

### Why strict parsing?

The defensive parser in earlier versions returned `is_injection=False` / `confidence=0.0` on any malformed shape — which silently produced false negatives. A prompt-injection detector that always says "no injection" is worse than one that errors out, because the caller can react to an error (retry, log, escalate) but can't react to silent false negatives.

If you see a `ToolError` in production, that means the engine's output drifted from the assumed schema. Fix the parser in the relevant tool file.

### Adjusting parser strictness

If you'd rather have lenient defaults for a specific tool (e.g., you trust the engine's output for that preset), edit the tool file in `decision_mcp/tools/` and replace the `choice_of` / `yes_prob` / `score_level` calls with manual extraction that returns defaults on missing keys. The shared helpers in `decision_mcp/tools/_helpers.py` are convenient but not mandatory.

---

## Adding a new tool

See `AGENTS.md` → "Common tasks → Add a new tool" for the canonical pattern. The summary:

1. Create `decision_mcp/tools/mytool.py` with `MyInput`, `MyOutput`, `MyTool(Tool)`.
2. Register in `decision_mcp/tools/__init__.py`.
3. Add a test in `tests/test_tools.py`.

That's it. Schema, route, and dispatch derive from the class.

## Adding a new engine preset

If the engine ships a new preset you want to expose:

1. Add a `PresetSpec` to the engine's `presets()` (e.g. in `decision_mcp/engines/laya.py`):
   ```python
   from laya.presets import my_new_questions
   presets["my_new_preset"] = PresetSpec(
       name="my_new_preset", questions=my_new_questions(), state_key="<field>"
   )
   ```
2. Create a tool file + register.

## Verifying tool output against the real engine

```python
import json
from decision_mcp.bridge import DecisionBridge

b = DecisionBridge(preload=True)   # needs the `laya` extra + ./models
raw = b.predict("some test input", preset="guard")
print(json.dumps(raw, indent=2, default=str))
```

Or run the opt-in integration tests, which exercise all the inference tools on the real models:

```bash
DECISIONMCP_INTEGRATION=1 pytest tests/test_integration.py
```

Use this to see the actual shape and adjust tool parsers. No mocks — runs against the real model.
