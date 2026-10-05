"""HTTP MCP server entry point.

Run with:

    uvicorn decision_mcp.server:app --host 127.0.0.1 --port 8765
    # or
    decisionmcp        # console script defined in pyproject.toml
    # or
    python -m decision_mcp.server

The MCP server is built on top of the official ``mcp[server]`` SDK and
exposes the registered tools from :mod:`decision_mcp.tools` over HTTP/SSE.

Error handling
--------------

Three error categories are handled distinctly:

1. **Unknown tool / invalid input** — protocol-level error returned to
   the caller. Doesn't log a traceback (just the message).
2. **Tool / bridge error** (any :class:`DecisionMCPError`) — the call is
   returned as MCP content with ``isError=true``. Logged with full
   traceback so operators can debug.
3. **Unexpected error** (anything else) — returned as a generic MCP
   error block ("Internal error. Check server logs."). Full traceback
   logged but **never** returned to the caller — avoid leaking
   internals (file paths, library names, host info).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from pydantic import ValidationError
from starlette.responses import Response
from starlette.routing import Mount

from .bridge import DecisionBridge
from .config import settings
from .errors import DecisionMCPError
from .tools import TOOLS
from .usage import UsageStore, current_session, current_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single bridge instance — engine + weights loaded once per process.
# ---------------------------------------------------------------------------
usage_store = (
    UsageStore(settings.data_dir / "usage.db", store_text=settings.usage_store_text)
    if settings.usage_enabled
    else None
)
bridge = DecisionBridge(preload=settings.preload_models, usage=usage_store)


# ---------------------------------------------------------------------------
# MCP server with dynamic tool registration from the TOOLS registry.
# ---------------------------------------------------------------------------
server: Server = Server("decisionmcp")


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """Return all registered tools to MCP clients."""
    return [types.Tool(**t.to_mcp_schema()) for t in TOOLS]


@server.call_tool()
async def _call_tool(name: str, arguments: dict) -> types.CallToolResult:
    """Dispatch an MCP tool call with full error handling.

    Errors are returned as ``CallToolResult(isError=True)`` (not raised) so the
    server stays alive and clients get a structured error to react to.
    """
    tool = next((t for t in TOOLS if t.name == name), None)
    if tool is None:
        available = [t.name for t in TOOLS]
        logger.warning("Unknown tool requested: %r (available: %s)", name, available)
        return _error_result(f"Unknown tool {name!r}. Available: {available}")

    # Validate inputs against the tool's Pydantic schema.
    try:
        validated = tool.input_schema(**arguments)
    except ValidationError as e:
        logger.warning("Tool %s got invalid input: %s", name, e)
        return _error_result(f"Invalid input for {name}: {e}")

    # Run the tool. Any DecisionMCPError is logged with traceback + returned
    # as MCP error block. Any other exception is treated as a bug —
    # logged with full traceback but a generic message returned.
    current_tool.set(name)  # lets the bridge attribute predictions to this tool
    try:
        result = await tool.run(bridge, **validated.model_dump())
        return _result(result)
    except DecisionMCPError as e:
        logger.error(
            "Tool %s failed (input_keys=%s): %s",
            name,
            sorted(arguments),
            e,
            exc_info=True,
        )
        return _error_result(str(e))
    except Exception:
        logger.exception("Unexpected error in tool %s", name)
        return _error_result(f"Internal error in {name}. Check server logs.")


# ---------------------------------------------------------------------------
# MCP wire-format helpers
# ---------------------------------------------------------------------------


def _result(result: Any) -> types.CallToolResult:
    """Format a successful Pydantic result as a JSON text ``CallToolResult``."""
    if hasattr(result, "model_dump_json"):
        text = result.model_dump_json()
    else:
        text = str(result)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


def _error_result(message: str) -> types.CallToolResult:
    """Format an error message as a ``CallToolResult`` with ``isError=True``."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)], isError=True
    )


# ---------------------------------------------------------------------------
# Plain FastAPI app with MCP SSE transport mounted manually + /health.
#
# Spec §9 / research 2026-09-22: mcp.server.fastapi.create_fastapi_app was
# removed in mcp 1.x and is broken on every released SDK today. Use the
# SSE transport directly on a plain FastAPI app. The /health endpoint
# is auth-exempt so the Docker healthcheck can probe TCP port-open
# ≈ models-resident (per spec §9 + the Phase 4 infra/smoke.sh contract).
# ---------------------------------------------------------------------------
sse = SseServerTransport("/messages/")
app = FastAPI(title="decisionmcp")


@app.get("/health")
async def health() -> dict[str, Any]:
    """TCP-port-open ≈ models-resident (models loaded at module import).

    spec §9: 'TCP healthcheck (port-open ≈ models-resident)' — if we
    answered, the models are loaded. No internal state to expose.
    """
    return {"status": "ok", "tools": len(TOOLS)}


@app.get("/sse")
async def sse_endpoint(request: Request) -> Response:
    """MCP SSE endpoint. Clients open this for the event stream.

    Each connection is one usage-log session; the id is set in a context var
    before ``server.run`` so request handlers spawned by it inherit it.

    The SSE transport writes the HTTP response itself, so this returns an
    empty ``Response`` at disconnect purely so FastAPI has nothing left to send
    (returning ``None`` makes it try, and uvicorn rejects the second response).
    """
    session_id = uuid.uuid4().hex
    current_session.set(session_id)
    if usage_store is not None:
        usage_store.open_session(session_id, request.headers.get("user-agent"))
    try:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    finally:
        if usage_store is not None:
            usage_store.close_session(session_id)
    return Response()


# MCP client → server POST endpoint (paired with /sse). Mounted as a raw ASGI app,
# not a FastAPI route: ``handle_post_message`` sends its own 202 response.
app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))


def main() -> None:
    """Console-script entry point: ``decisionmcp``."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "Starting DecisionMCP on http://%s:%d (engine=%s, tools=%d)",
        settings.host,
        settings.port,
        bridge.engine.name,
        len(TOOLS),
    )
    uvicorn.run(
        "decision_mcp.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
