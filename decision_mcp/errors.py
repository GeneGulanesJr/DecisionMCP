"""Custom exceptions for DecisionMCP.

All exceptions inherit from :class:`DecisionMCPError` so the server can catch
one base class and turn any project-specific failure into an MCP error
response — without leaking internals to the caller.
"""
from __future__ import annotations


class DecisionMCPError(Exception):
    """Base class for all DecisionMCP-specific errors."""


# ---------------------------------------------------------------------------
# Bridge errors — the active decision engine failed
# ---------------------------------------------------------------------------


class BridgeError(DecisionMCPError):
    """A call into the decision engine failed.

    The original exception is attached via ``__cause__`` for traceback
    chaining. The ``message`` includes enough context (which preset,
    approximate input size) to debug without exposing user data.
    """


class ModelLoadError(BridgeError):
    """Failed to load model weights (download, OOM, missing checkpoint, etc.).

    Raised at :class:`decision_mcp.bridge.DecisionBridge` construction time
    if ``preload=True``.
    """


class UnknownPresetError(BridgeError):
    """A tool requested a preset name that's not in :attr:`DecisionBridge.presets`."""


# ---------------------------------------------------------------------------
# Tool errors — tool-level parsing/execution failed
# ---------------------------------------------------------------------------


class ToolError(DecisionMCPError):
    """A tool failed to parse or execute.

    Attributes:
        tool_name: Name of the tool that failed.
        cause: Original exception if any (else ``None``).
    """

    def __init__(self, tool_name: str, message: str, *, cause: Exception | None = None) -> None:
        self.tool_name = tool_name
        self.cause = cause
        full = f"[{tool_name}] {message}"
        if cause is not None:
            full = f"{full}: {cause}"
        super().__init__(full)
        # Chain explicitly so callers that construct ToolError(cause=...) without
        # `raise ... from` still get the original in tracebacks.
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Server / protocol errors
# ---------------------------------------------------------------------------


class ServerError(DecisionMCPError):
    """Generic server-level error (used for unknown-tool, malformed request)."""
