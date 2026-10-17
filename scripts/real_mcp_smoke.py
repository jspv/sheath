"""Manual real-MCP smoke test (spec §12 gate).

Connects the sheath, via its normal public API, to a real stdio MCP server
(the local msgraph-email-calendar-mcp) and runs an agent task that pulls a
sizeable payload. We instrument sheath.spill._maybe_spill to record exactly
what type/size MAF hands our result_parser for a real MCP return -- the open
question the gate exists to answer -- and report whether anything spilled.
"""

from __future__ import annotations

import asyncio
import os
import sys

from agent_framework import MCPStdioTool

import sheath.spill as spill
from sheath import Sheath
from sheath.config import SheathConfig

# Directory of a stdio MCP server to test against. Override via argv[1] or $MCP_SERVER_DIR;
# defaults to the sibling msgraph server this gate was first run against.
MSGRAPH_DIR = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("MCP_SERVER_DIR", "../msgraph-email-calendar-mcp")
)

# --- instrument the spill decision point to observe real MCP returns ----------
seen: list[dict] = []
_orig = spill._maybe_spill


def _spy(session, tool_name, result):
    import json

    size = None
    try:
        if isinstance(result, (bytes, bytearray, str)):
            size = len(result if isinstance(result, str) else result)
        elif isinstance(result, (dict, list)):
            size = len(json.dumps(result, default=str).encode())
    except Exception:
        size = "?"
    out = _orig(session, tool_name, result)
    spilled = spill._is_handle_summary(out)
    seen.append({
        "tool": tool_name,
        "raw_type": type(result).__name__,
        "approx_size": size,
        "spilled": spilled,
    })
    return out


spill._maybe_spill = _spy


async def main() -> None:
    mcp = MCPStdioTool(
        name="msgraph",
        command="sh",
        args=["-c", f"cd {MSGRAPH_DIR} && exec uv run msgraph-mcp"],
    )

    h = Sheath(SheathConfig(model="gpt-6.1-sol"), bundles=("code", "files"))
    result = await h.asolve(
        "Using the msgraph tools, fetch my single most recent email INCLUDING its "
        "full body, then report only its subject, sender, and date. Do not summarize "
        "the body.",
        tools=[mcp],
    )

    print("\n========== RESULT ==========")
    print("final_text:\n", result.final_text)
    print("error:", result.error)
    print("session_dir:", result.session_dir)
    print("handles:", list(result.handles.keys()))
    for hid, meta in result.handles.items():
        print("  handle", hid, "->", meta)

    print("\n========== SPILL PARSER OBSERVATIONS ==========")
    if not seen:
        print("  (result_parser was never invoked on an MCP return!)")
    for row in seen:
        print(" ", row)


if __name__ == "__main__":
    asyncio.run(main())
