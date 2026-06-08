"""MCP connection seam — connect the agent to external tools via MCP.

The Google for Startups challenge highlights the **Model Context Protocol (MCP)**
as the way agents "securely connect to external tools, gather context, and
execute tasks." This module is the seam where the Anukriti agent connects to a
*remote* deterministic PGx engine over MCP instead of using the bundled offline
tables.

DESIGN: off-by-default. The hackathon submission runs fully offline (so judges
get byte-identical results with zero provisioning). When `ANUKRITI_MCP_URL` is
set, `build_mcp_toolset()` returns an ADK MCPToolset pointed at that server, and
the agent transparently gains the remote engine's tools. The runtime guardrail
in `enforcement.py` applies identically to remote tool results — the boundary
does not care whether the engine ran locally or over MCP.

This keeps the architecture honest: the optimization (the guardrail + the
named-refusal contract) is independent of where the deterministic truth comes
from, which is exactly what makes it production-portable.
"""

from __future__ import annotations

import os
from typing import Any


def mcp_enabled() -> bool:
    """True when an external MCP engine URL is configured."""
    return bool(os.environ.get("ANUKRITI_MCP_URL", "").strip())


def mcp_config() -> dict:
    """Report the current MCP wiring (safe to log — no secrets)."""
    return {
        "enabled": mcp_enabled(),
        "url": os.environ.get("ANUKRITI_MCP_URL", ""),
        "transport": os.environ.get("ANUKRITI_MCP_TRANSPORT", "sse"),
        "auth": "header-bearer" if os.environ.get("ANUKRITI_MCP_TOKEN") else "none",
    }


def build_mcp_toolset() -> Any | None:
    """Return an ADK MCPToolset for the remote PGx engine, or None if disabled.

    The remote server is expected to expose the same three tools the offline
    layer provides (`check_pgx_safety`, `search_pgx_evidence`,
    `list_supported_workflows`) with identical structured results, so the agent
    and the guardrail are unchanged whether tools are local or remote.

    Auth: if `ANUKRITI_MCP_TOKEN` is set, it is sent as a Bearer header. Never
    embed credentials in code or commit them — this reads from the environment.
    """
    if not mcp_enabled():
        return None

    url = os.environ["ANUKRITI_MCP_URL"].strip()
    transport = os.environ.get("ANUKRITI_MCP_TRANSPORT", "sse").strip().lower()
    token = os.environ.get("ANUKRITI_MCP_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None

    # Import locally so the package imports cleanly when MCP is not used and the
    # optional connection params API surface differs across ADK point releases.
    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

        if transport == "sse":
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                SseConnectionParams,
            )
            conn = SseConnectionParams(url=url, headers=headers)
        else:
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                StreamableHTTPConnectionParams,
            )
            conn = StreamableHTTPConnectionParams(url=url, headers=headers)

        return MCPToolset(connection_params=conn)
    except Exception as exc:  # pragma: no cover - depends on optional remote
        # Fail visible-but-safe: surface why MCP didn't attach, keep offline tools.
        import sys
        print(
            f"[anukriti-adk] MCP toolset unavailable ({exc!r}); "
            "falling back to offline-deterministic tools.",
            file=sys.stderr,
        )
        return None
