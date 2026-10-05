/**
 * memory-save-classified — atomic LaPis memory-save + DecisionMCP classification.
 *
 * USAGE
 * -----
 * Drop this file into your LaPis repo (e.g. `src/tools/memory-save-classified.ts`),
 * adjust the imports at the top to match your existing tool-registry pattern,
 * then register `memorySaveClassifiedTool` like any other tool.
 *
 * If you keep this file in DecisionMCP/integrations/ instead, copy the export
 * into LaPis when ready to wire it up.
 *
 * WHAT IT DOES
 * ------------
 * Same shape as `memory-save`, but classifies the content via DecisionMCP first
 * and applies safety rules. Refuses prompt injections by default.
 *
 * CONFIG (env, all LAPIS_* prefixed)
 * ----------------------------------
 *   LAPIS_DECISIONMCP_URL=http://127.0.0.1:8765      # default
 *   LAPIS_DECISIONMCP_ENABLED=true                  # default; set "false" to bypass
 *   LAPIS_DECISIONMCP_TIMEOUT_MS=5000               # default
 *
 * FAILURE SEMANTICS (defensive — default to safe)
 * -----------------------------------------------
 *   decision_guard fails + on_injection="refuse" (default)
 *     → throw, don't save
 *   decision_guard fails + on_injection="save_as_security_block"
 *     → save as type="bugfix", topic_key="security-blocks"
 *   decision_moderate fails
 *     → save as type="bugfix", topic_key="moderation-blocks"
 *   decision_triage / decision_email fails
 *     → save without classification metadata
 *   DecisionMCP unreachable (network error, 5xx)
 *     → same as above per tool, plus warning log
 *
 * TRUST SCORING
 * -------------
 *   final_trust = base_trust * engine_confidence
 *   High engine confidence → high trust → survives LaPis dedup/expire cycles.
 *   Low confidence injection → save as warning, trust_score=0.2.
 */

// ============================================================================
// Types — adapt to your existing LaPis type definitions if they already exist.
// ============================================================================

export type MemoryType =
  | "decision"
  | "bugfix"
  | "pattern"
  | "discovery"
  | "config"
  | "preference"
  | "learning";

export interface MemorySaveInput {
  title: string;
  content: string;
  type?: MemoryType;
  topic_key?: string;
  trust_score?: number;
  expires_in?: string;
}

export interface MemorySaveResult {
  id: string;
  title: string;
  content: string;
  type: MemoryType;
  topic_key?: string;
  trust_score: number;
  created_at: string;
}

export type ClassificationTool = "guard" | "moderate" | "triage" | "email" | "auto";

export type OnInjectionBehavior = "refuse" | "save_as_security_block";

export interface MemorySaveClassifiedInput extends MemorySaveInput {
  /** Which DecisionMCP tool to use. "auto" = heuristic dispatch. Default: "auto". */
  classification?: ClassificationTool;
  /** What to do if decision_guard detects injection. Default: "refuse". */
  on_injection?: OnInjectionBehavior;
}

export interface ClassificationMetadata {
  /** Which classification tool ran. */
  tool: ClassificationTool;
  /** Tool-specific result (is_injection, tier, intent, etc.). */
  result: Record<string, unknown>;
  /** The engine's confidence (0-1). */
  confidence: number;
  /** When classification happened (ISO timestamp). */
  classified_at: string;
}

/**
 * Context provided by the LaPis host. Adapt to your handler signature.
 * - saveMemory: the existing memory-save handler
 * - log: structured logger (or pass console / a no-op for tests)
 */
export interface ToolContext {
  saveMemory: (input: MemorySaveInput) => Promise<MemorySaveResult>;
  log: (
    level: "info" | "warn" | "error",
    msg: string,
    meta?: Record<string, unknown>,
  ) => void;
}

// ============================================================================
// Config
// ============================================================================

interface Config {
  decisionmcpUrl: string;
  enabled: boolean;
  timeoutMs: number;
}

function loadConfig(): Config {
  return {
    decisionmcpUrl: process.env.LAPIS_DECISIONMCP_URL ?? "http://127.0.0.1:8765",
    enabled: (process.env.LAPIS_DECISIONMCP_ENABLED ?? "true").toLowerCase() !== "false",
    timeoutMs: parseInt(process.env.LAPIS_DECISIONMCP_TIMEOUT_MS ?? "5000", 10),
  };
}

// ============================================================================
// DecisionMCP HTTP client (JSON-RPC 2.0 over HTTP)
// ============================================================================

class DecisionMCPError extends Error {
  constructor(
    message: string,
    public readonly tool: ClassificationTool,
  ) {
    super(message);
    this.name = "DecisionMCPError";
  }
}

async function callDecisionMCP(
  tool: ClassificationTool,
  state: string,
  cfg: Config,
): Promise<ClassificationMetadata> {
  if (!cfg.enabled) {
    throw new DecisionMCPError("DecisionMCP disabled via LAPIS_DECISIONMCP_ENABLED=false", tool);
  }

  const toolName = {
    guard: "decision_guard",
    moderate: "decision_moderate",
    triage: "decision_triage",
    email: "decision_email",
  }[tool] as string;

  const args =
    tool === "guard" ? { prompt: state } :
    tool === "moderate" ? { text: state } :
    tool === "triage" ? { text: state } :
    tool === "email" ? { body: state } :
    { prompt: state };

  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), cfg.timeoutMs);

  try {
    const response = await fetch(`${cfg.decisionmcpUrl}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name: toolName, arguments: args },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new DecisionMCPError(`HTTP ${response.status} ${response.statusText}`, tool);
    }

    const json = (await response.json()) as {
      result?: { content: Array<{ type: string; text: string; isError?: boolean }> };
      error?: { message: string };
    };

    if (json.error) {
      throw new DecisionMCPError(json.error.message, tool);
    }

    const block = json.result?.content?.[0];
    if (!block) {
      throw new DecisionMCPError("empty response from DecisionMCP", tool);
    }
    if (block.isError) {
      throw new DecisionMCPError(block.text, tool);
    }

    const result = JSON.parse(block.text) as Record<string, unknown>;
    const confidence =
      typeof result.confidence === "number" ? result.confidence : 0.5;

    return {
      tool,
      result,
      confidence,
      classified_at: new Date().toISOString(),
    };
  } catch (e) {
    if (e instanceof DecisionMCPError) throw e;
    throw new DecisionMCPError(e instanceof Error ? e.message : String(e), tool);
  } finally {
    clearTimeout(timeoutHandle);
  }
}

// ============================================================================
// Classification dispatch (auto mode)
// Heuristics are conservative — when in doubt, classify. Caller can override.
// ============================================================================

function autoClassify(content: string): ClassificationTool | null {
  // Email — has From: or Subject: header
  if (/^from:\s\S/im.test(content) || /^subject:\s\S/im.test(content)) {
    return "email";
  }

  // Ticket — has Ticket # or ticket-specific keywords
  if (
    /^ticket\s*#?\d/im.test(content) ||
    /\b(status|priority|customer[\s_]impact)\b/i.test(content)
  ) {
    return "triage";
  }

  // Short, single-paragraph → likely a prompt
  if (content.length < 500 && !/\n\s*\n/.test(content)) {
    return "guard";
  }

  // Multi-paragraph, longer → likely post / reply
  if (content.length >= 500) {
    return "moderate";
  }

  // No clear signal — skip classification
  return null;
}

// ============================================================================
// Main tool implementation
// ============================================================================

export async function memorySaveClassified(
  input: MemorySaveClassifiedInput,
  ctx: ToolContext,
): Promise<MemorySaveResult & { classification?: ClassificationMetadata }> {
  const cfg = loadConfig();
  const classification = input.classification ?? "auto";
  const onInjection = input.on_injection ?? "refuse";

  // Resolve which classification tool to use
  const classifierTool: ClassificationTool | null =
    classification === "auto" ? autoClassify(input.content) : classification;

  // No classification needed → save directly
  if (classifierTool === null) {
    return await ctx.saveMemory(input);
  }

  // Try to classify
  let metadata: ClassificationMetadata | null = null;
  try {
    metadata = await callDecisionMCP(classifierTool, input.content, cfg);
  } catch (e) {
    const err = e as DecisionMCPError;
    ctx.log("warn", `DecisionMCP ${err.tool} failed: ${err.message}`, { tool: err.tool });

    // Defensive defaults per tool
    if (classifierTool === "guard") {
      if (onInjection === "refuse") {
        throw new Error(
          `Refusing to save: decision_guard failed and on_injection="refuse". ` +
            `Original error: ${err.message}. ` +
            `Use on_injection="save_as_security_block" to override.`,
        );
      }
      // save_as_security_block
      return await ctx.saveMemory({
        ...input,
        type: "bugfix",
        topic_key: "security-blocks",
        content: `${input.content}\n\n[Guard check failed: ${err.message}]`,
        trust_score: 0.1,
      });
    }

    if (classifierTool === "moderate") {
      return await ctx.saveMemory({
        ...input,
        type: "bugfix",
        topic_key: "moderation-blocks",
        content: `${input.content}\n\n[Moderate check failed: ${err.message}]`,
        trust_score: 0.1,
      });
    }

    // triage / email failure — save without classification
    return await ctx.saveMemory(input);
  }

  // Classification succeeded — apply safety logic
  if (classifierTool === "guard" && metadata.result.is_injection === true) {
    if (onInjection === "refuse") {
      throw new Error(
        `Refusing to save: prompt injection detected ` +
          `(confidence=${metadata.confidence}). ` +
          `Use on_injection="save_as_security_block" to override.`,
      );
    }
    return await ctx.saveMemory({
      ...input,
      type: "bugfix",
      topic_key: "security-blocks",
      content:
        `${input.content}\n\n` +
        `[Blocked: prompt injection detected, confidence=${metadata.confidence}]`,
      trust_score: 0.2,
    });
  }

  // Save with classification metadata + engine-confidence-weighted trust
  const baseTrust = input.trust_score ?? 1.0;
  const adjustedTrust = baseTrust * metadata.confidence;

  const result = await ctx.saveMemory({
    ...input,
    trust_score: adjustedTrust,
  });

  return { ...result, classification: metadata };
}

// ============================================================================
// Tool registration
// ============================================================================

export const memorySaveClassifiedTool = {
  name: "memory-save-classified",
  description:
    "Save a memory to LaPis with DecisionMCP classification. " +
    "Auto-detects content type (prompt / email / ticket / post) and runs the " +
    "appropriate DecisionMCP tool (decision_guard, decision_moderate, decision_triage, decision_email). " +
    "Refuses to save prompt injections by default (overridable via on_injection). " +
    "Returns the saved memory plus classification metadata.",
  inputSchema: {
    type: "object",
    properties: {
      title: {
        type: "string",
        description: "Memory title.",
      },
      content: {
        type: "string",
        description: "Memory content. Will be classified via DecisionMCP.",
      },
      type: {
        type: "string",
        enum: ["decision", "bugfix", "pattern", "discovery", "config", "preference", "learning"],
        description: "Memory type. Default: 'pattern'.",
      },
      topic_key: {
        type: "string",
        description: "Topic key for grouping (use standardized keys — see SKILL.md).",
      },
      classification: {
        type: "string",
        enum: ["guard", "moderate", "triage", "email", "auto"],
        description: "Which DecisionMCP tool to use. Default: 'auto' (heuristic dispatch).",
      },
      on_injection: {
        type: "string",
        enum: ["refuse", "save_as_security_block"],
        description: "What to do if decision_guard detects injection. Default: 'refuse'.",
      },
      trust_score: {
        type: "number",
        description:
          "Base trust score (0-1). Multiplied by engine confidence before saving. " +
          "Default: 1.0.",
      },
      expires_in: {
        type: "string",
        description: "TTL (e.g., '7d', '30d', '12h').",
      },
    },
    required: ["title", "content"],
  },
  handler: memorySaveClassified,
};

/**
 * Suggested tests to add (use vitest / node:test):
 *
 * 1. autoClassify detects email / ticket / short-prompt / long-post / ambiguous
 * 2. memorySaveClassified refuses prompt injection (is_injection=true, default on_injection)
 * 3. memorySaveClassified saves injection as security-block when on_injection="save_as_security_block"
 * 4. memorySaveClassified defaults to refusing on decision_guard failure
 * 5. memorySaveClassified saves as moderation-block on decision_moderate failure
 * 6. memorySaveClassified saves without classification on decision_triage failure
 * 7. memorySaveClassified applies trust_score = base * engine_confidence
 * 8. memorySaveClassified skips classification when content is ambiguous and classification="auto"
 * 9. memorySaveClassified respects explicit classification= override
 * 10. callDecisionMCP times out after LAPIS_DECISIONMCP_TIMEOUT_MS
 * 11. callDecisionMCP returns DecisionMCPError on HTTP 5xx
 * 12. callDecisionMCP returns DecisionMCPError on JSON-RPC error response
 */