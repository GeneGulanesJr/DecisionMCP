"""Transport-level tests for the HTTP server (mocked, no models needed).

Covers what the tool/bridge tests can't: the FastAPI app's wiring —
``/health``, legacy SSE (``/sse``) and the streamable-HTTP endpoint
(``/mcp``) that pi's built-in MCP client speaks.

``decision_mcp.server`` builds the bridge and usage store at import time,
so the fixture patches settings (tmp data dir, no preload) *before* the
first import of the module.
"""
from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0.0.0"},
    },
}

POST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """One app run per module — StreamableHTTPSessionManager.run() is
    once-per-instance, so the lifespan must start exactly once per process.
    """
    import decision_mcp.config as cfg

    saved = (cfg.settings.data_dir, cfg.settings.preload_models)
    cfg.settings.data_dir = tmp_path_factory.mktemp("decisionmcp-data")
    cfg.settings.preload_models = False
    try:
        import decision_mcp.server as srv

        with TestClient(srv.app) as c:  # context manager runs the lifespan
            yield c
    finally:
        cfg.settings.data_dir, cfg.settings.preload_models = saved


def _rpc(
    method: str, params: dict[str, Any] | None = None, id_: int | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if id_ is not None:
        msg["id"] = id_
    return msg


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tools"] == 13


def test_streamable_initialize_and_list_tools(client: TestClient) -> None:
    r = client.post("/mcp", json=INIT, headers=POST_HEADERS)
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == "decisionmcp"
    session = r.headers["mcp-session-id"]

    # lifecycle notification — accepted, no body
    r = client.post(
        "/mcp",
        json=_rpc("notifications/initialized"),
        headers={**POST_HEADERS, "mcp-session-id": session},
    )
    assert r.status_code in (200, 202)

    r = client.post(
        "/mcp",
        json=_rpc("tools/list", id_=2),
        headers={**POST_HEADERS, "mcp-session-id": session},
    )
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert len(names) == 13
    assert "decision_guard" in names


def test_streamable_unknown_tool_is_error_result(client: TestClient) -> None:
    r = client.post("/mcp", json=INIT, headers=POST_HEADERS)
    session = r.headers["mcp-session-id"]

    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call", params={"name": "decision_nope", "arguments": {}}, id_=3
        ),
        headers={**POST_HEADERS, "mcp-session-id": session},
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True
    assert "decision_nope" in result["content"][0]["text"]


def test_streamable_delete_terminates_session(client: TestClient) -> None:
    r = client.post("/mcp", json=INIT, headers=POST_HEADERS)
    session = r.headers["mcp-session-id"]
    r = client.delete("/mcp", headers={"mcp-session-id": session})
    assert r.status_code in (200, 204)
    # a terminated session must be rejected afterwards
    r = client.post(
        "/mcp",
        json=_rpc("tools/list", id_=4),
        headers={**POST_HEADERS, "mcp-session-id": session},
    )
    assert r.status_code == 404


def test_sse_legacy_route_wired(client: TestClient) -> None:
    # NOTE: the SSE handshake itself can't be exercised through TestClient
    # (it runs the ASGI app to completion — an infinite SSE stream hangs).
    # Verified live in scripts/smoke against a running server.
    from starlette.routing import Mount

    import decision_mcp.server as srv

    paths = {getattr(r, "path", None) for r in srv.app.routes}
    assert {"/sse", "/messages", "/health"} <= paths
    # /mcp itself is served by the root-mounted fallback guard (last route):
    # Starlette 1.7's Mount("/mcp") 307-redirects the exact path under real
    # ASGI servers, so the transport is path-guarded inside Mount("/") instead.
    last = srv.app.routes[-1]
    assert isinstance(last, Mount)
    assert isinstance(last.app, srv._StreamableHTTPGuard)
