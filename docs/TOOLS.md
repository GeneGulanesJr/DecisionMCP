# Tools reference

Eleven tools run inference through `LayaBridge.predict(state, preset)` or `LayaBridge.predict_custom(state, questions, state_key)`. Each wraps a specific upstream Laya preset (or custom question set) and parses the result into a typed Pydantic output schema. Two more (`laya_update`, `laya_usage`) are maintenance tools that don't run the models.

## How Laya answers

`Router.predict` returns `{"answers": {<question>: <answer>}, "usage": ..., "routing": ...}`. Each answer is one of three typed shapes, and every tool parses them with the helpers in `laya_mcp/tools/_helpers.py`:

| Question type | Answer field | Meaning |
| ------------- | ------------ | ------- |
| `choice` | `choice` | The winning label, plus per-label `probabilities` |
| `noul` | `noul` | Probability of "yes" (tools threshold at 0.5) |
| `score` | `score` + `legend` | Expected level `0..N-1` with a text legend, plus per-level `probabilities` |

Every answer also has `answer_confidence` in `[0, 1]`, which is what the tools report as `confidence`. `details` in each tool's output is the full raw Laya result (including which checkpoint answered).

The text is passed to Laya as `{<field>: text}` because each question's instructions refer to its input by name (`prompt`, `request`, `post`, `message`, `body` for the presets; see `LayaBridge.PRESET_STATE_KEYS`).

## laya_guard

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
    details: dict        # raw Laya response
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

**Where the parser lives:** `laya_mcp/tools/guard.py`

---

## laya_route

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

**How it decides:** Laya's `difficulty` question has four levels (trivial / easy / moderate / hard). The tier is `frontier` when at least 50% of the probability mass is on moderate + hard, otherwise `small`. `confidence` is the mass on the side chosen. To change the cut-off, edit `_FRONTIER_LEVELS` in `laya_mcp/tools/route.py`.

**Cost:** ~33 ms (T4). Saves potentially hundreds of milliseconds + dollars per call when routing to a small model.

**Where the parser lives:** `laya_mcp/tools/route.py`

---

## laya_triage

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

**Upstream questions used:** `intent` (choice), `is_urgent` / `refund_requested` / `churn_risk` (noul), `frustration` (score, four levels; the legend text is returned). These come from `laya.presets.triage_questions`; if a Laya upgrade renames them the tool raises `ToolError` naming the missing key.

**Where the parser lives:** `laya_mcp/tools/triage.py`

---

## laya_moderate

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

`details` also carries Laya's `spam` and `severity` answers.

**When to use:** Before any user-facing output is shown. Run on:
- Agent responses that get posted to public channels
- User-submitted content in community spaces
- Inbound messages from unknown senders

**Limitations:**
- Not a substitute for human moderation.
- False positives possible (sarcasm, reclaimed slurs, in-group speech).
- False negatives possible (especially for novel attacks or coded language).

**Where the parser lives:** `laya_mcp/tools/moderate.py`

---

## laya_email

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

**How it decides:** the boolean fields are P(yes) ≥ 0.5 on Laya's `needs_reply`, `is_spam` and `is_phishing` questions.

**Cost:** ~33 ms.

**Where the parser lives:** `laya_mcp/tools/email.py`

---

## Coding-specific tools

The six tools below use **custom Laya question schemas** (not upstream presets) and are tuned for coding workflows. They all go through `LayaBridge.predict_custom()` — `bridge.predict()` is reserved for the five upstream presets above.

Each defines its questions as a plain dict at the top of its file (`{"type": "choice", "instructions": "... `text` ...", "criteria": {label: description}}`). Two things learned while tuning them against the real models:

- **Use `choice` for ordinal scales too** (low / medium / high, S0–S3, skip → critical). Laya's `score` type gave noticeably worse answers on these custom questions (e.g. a `nit` comment rated medium priority, a README typo rated S2); the same scales as `choice` fixed most of them.
- **Custom questions are not what Laya was trained on**, so treat `confidence` on them as a rough signal. Coarse labels (tone, type, area, kind of secret) are reliable; fine-grained ones (`scope`, `risk`) are less so. `tests/test_integration.py` shows what is verified.

---

## laya_review_tone

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

**Where the parser lives:** `laya_mcp/tools/review_tone.py`

---

## laya_bug_severity

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

**Where the parser lives:** `laya_mcp/tools/bug_severity.py`

---

## laya_commit_classify

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

**Where the parser lives:** `laya_mcp/tools/commit_classify.py`

---

## laya_test_priority

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

**Where the parser lives:** `laya_mcp/tools/test_priority.py`

---

## laya_secret_risk

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

**Different from `laya_guard`:** `laya_guard` detects prompt-injection (attempts to manipulate the agent). `laya_secret_risk` detects leaked credentials (data exposure). They're separate concerns — run both.

**Where the parser lives:** `laya_mcp/tools/secret_risk.py`

---

## laya_diff_intent

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

**Where the parser lives:** `laya_mcp/tools/diff_intent.py`

---

## Maintenance tools

These two don't run the models.

---

## laya_update

Check whether the Laya library or its model weights are out of date, and whether new Laya models exist. Laya has no update mechanism of its own.

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

**What `check` compares:**
- installed `laya` vs. the latest release on PyPI
- the commit of each model repo in the local cache (`./models`) vs. the Hub head
- new models: extra top-level model folders in `convaiinnovations/laya` (each has an `rl_agent_config.json`) and any other `convaiinnovations/laya*` repo that this laya version doesn't list. New models are only reported, never downloaded, because an older `laya` may not be able to load them; upgrade first.

**`apply`:** runs `pip install --upgrade laya` (via `uv pip` when available) and re-downloads model repos whose commit changed. It only ever installs the `laya` package. Because it runs pip and the HTTP server has no auth, it is **refused unless `LAYAMCP_ALLOW_UPDATES=true`**. Nothing takes effect in the running process, so `restart_required` is set: restart `layamcp` afterwards, then run `check` again (a new laya may list new models).

Network failures (PyPI or the Hub unreachable) raise `ToolError` rather than returning a partial answer.

**Where it lives:** `laya_mcp/tools/update.py`

---

## laya_usage

Summarise what Laya has been used for, or export it as training data. Laya only exposes per-call hooks (`run_id`, token usage, timing) and persists nothing, so LayaMCP keeps its own log.

**Input:** `action`: `"stats"` (default) or `"export"`; optional `since_hours`; for `export`, optional `labeled_only` (only rows that carry input text — ready `(input, answers)` training pairs; the export file is then named `usage-labeled-<timestamp>.jsonl`).

**What is logged:** every prediction, success or failure, in a SQLite file (`<data_dir>/usage.db`, default `./data/usage.db`): timestamp, MCP session id, tool, preset, checkpoint that answered, latency, input length and tokens, Laya's full answers (labels + probabilities) and routing. One `sessions` row per MCP connection records the client's user-agent and when it disconnected.

**Input text is NOT stored by default**, because inputs can hold prompts, source code and credentials. Set `LAYAMCP_USAGE_STORE_TEXT=true` to keep it; that is what makes an export usable as `(input, answers)` pairs. Even then, text sent to `laya_secret_risk` is never stored.

**`export`** writes successful calls as JSONL to `<data_dir>/exports/usage-<timestamp>.jsonl` and returns the path and row count (the file path is never caller-supplied). Each line: `{run_id, ts, session_id, tool, preset, model, input, answers}`, where `input` is `null` for calls made while text storage was off. Pass `labeled_only=true` to skip those rows entirely.

The Laya answers are the model's own predictions, not ground truth. For supervised training you will still need to review or correct them.

Disable logging with `LAYAMCP_USAGE_ENABLED=false` (the tool then returns an error).

**Where it lives:** `laya_mcp/tools/usage_report.py` (tool), `laya_mcp/usage.py` (store)

---

## Error handling

Tools **raise on any unexpected Laya output** rather than returning silent defaults. The server catches these and returns them as MCP error content blocks (with `isError=true`).

### Exception hierarchy

```
LayaMCPError            (catch this for any project-specific failure)
├── ModelLoadError       (Laya weights couldn't load)
├── UnknownPresetError  (tool requested a preset that doesn't exist)
└── BridgeError         (upstream Laya call failed: GPU OOM, runtime error, etc.)
```

`ToolError` (also extends `LayaMCPError`) is raised when a tool's parser can't extract what it needs from Laya's response. It carries the tool name and an optional `cause`.

### When a tool raises

| Failure                                        | Raised by      | Server response to client                  |
| ---------------------------------------------- | -------------- | ------------------------------------------ |
| Upstream Laya raises (GPU OOM, runtime error)  | `BridgeError`  | MCP error block: `"[laya_guard] Laya predict failed (preset='guard', state_len=12)"` |
| Tool requested a preset not in `LayaBridge.PRESETS` | `UnknownPresetError` | MCP error block: `"[laya_guard] Unknown preset 'bogus'. Choose from [...]"` |
| Laya returned a non-dict                       | `ToolError`    | MCP error block: `"[laya_guard] Expected dict from Laya, got str: 'oops'"` |
| Laya returned an empty dict                    | `ToolError`    | MCP error block: `"[laya_guard] Laya returned an empty result."` |
| Expected question missing in Laya output       | `ToolError`    | MCP error block: `"[laya_guard] Missing expected key 'jailbreak' in Laya answers. Got keys: []"` |
| Answer has the wrong type for the question     | `ToolError`    | MCP error block: `"[laya_guard] Expected 'noul' answer at 'jailbreak', got type 'choice'"` |
| Confidence isn't numeric                       | `ToolError`    | MCP error block: `"[laya_guard] Confidence at 'jailbreak' is not numeric: 'NaN'"` |
| PyPI / Hugging Face Hub unreachable (`laya_update`) | `ToolError` | MCP error block: `"[laya_update] Could not query the Hugging Face Hub: ..."` |
| `laya_update` `apply` while updates are disabled | `ToolError` | MCP error block: `"[laya_update] Updates are disabled. Set LAYAMCP_ALLOW_UPDATES=true ..."` |
| Tool input fails schema validation             | `ValidationError` (Pydantic) | MCP error block: `"Invalid input for laya_guard: ..."` |
| Unknown tool name                              | (server-side)  | MCP error block: `"Unknown tool 'foo'. Available: ['laya_guard', ...]"` |

### What the server logs

All `LayaMCPError`s are logged with **full traceback** at `ERROR` level, including the input keys (never values — to avoid leaking prompts):

```
2024-01-15 12:34:56 ERROR laya_mcp.server Tool laya_guard failed (input_keys=['prompt']): [laya_guard] Expected dict from Laya, got str: 'oops'
Traceback (most recent call last):
  ...
```

Unexpected exceptions (not `LayaMCPError`) are also logged with traceback, but the **caller sees only** `"Internal error in laya_guard. Check server logs."` to prevent leaking internals.

### Why strict parsing?

The defensive parser in earlier versions returned `is_injection=False` / `confidence=0.0` on any malformed shape — which silently produced false negatives. A prompt-injection detector that always says "no injection" is worse than one that errors out, because the caller can react to an error (retry, log, escalate) but can't react to silent false negatives.

If you see a `ToolError` in production, that means Laya's output drifted from the assumed schema. Fix the parser in the relevant tool file.

### Adjusting parser strictness

If you'd rather have lenient defaults for a specific tool (e.g., you trust Laya's output for that preset), edit the tool file in `laya_mcp/tools/` and replace the `choice_of` / `yes_prob` / `score_level` calls with manual extraction that returns defaults on missing keys. The shared helpers in `laya_mcp/tools/_helpers.py` are convenient but not mandatory.

---

## Adding a new tool

See `AGENTS.md` → "Common tasks → Add a new tool" for the canonical pattern. The summary:

1. Create `laya_mcp/tools/mytool.py` with `MyInput`, `MyOutput`, `MyTool(Tool)`.
2. Register in `laya_mcp/tools/__init__.py`.
3. Add a test in `tests/test_tools.py`.

That's it. Schema, route, and dispatch derive from the class.

## Adding a new upstream preset

If Laya ships a new preset you want to expose:

1. Add it to `LayaBridge.PRESETS` in `laya_mcp/bridge.py`:
   ```python
   from laya.presets import my_new_questions
   PRESETS["my_new_preset"] = my_new_questions
   ```
2. Create a tool file + register.

## Verifying tool output against real Laya

```python
import json
from laya_mcp.bridge import LayaBridge

b = LayaBridge(preload=True)
raw = b.predict("some test input", preset="guard")
print(json.dumps(raw, indent=2, default=str))
```

Or run the opt-in integration tests, which exercise all the inference tools on the real models:

```bash
LAYAMCP_INTEGRATION=1 pytest tests/test_integration.py
```

Use this to see the actual shape and adjust tool parsers. No mocks — runs against the real model.