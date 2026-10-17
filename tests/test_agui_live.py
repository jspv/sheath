import importlib.util
import os
import sys
from pathlib import Path

import pytest

_RUN = os.environ.get("SHEATH_LIVE_AGUI") == "1"
_HAS_AGUI = importlib.util.find_spec("agent_framework_ag_ui") is not None
pytestmark = pytest.mark.skipif(
    not (_RUN and _HAS_AGUI),
    reason="set SHEATH_LIVE_AGUI=1 and install the 'agui' extra to run AG-UI live tests",
)

_FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_progress_server.py")


def _event_types(events):
    return [type(e).__name__ for e in events]


def _collect(input_data, **sheath_kwargs):
    import asyncio

    from sheath import Sheath, SheathConfig

    async def run():
        h = Sheath(SheathConfig(), **sheath_kwargs)
        return [ev async for ev in h.agui_stream(input_data)]

    return asyncio.run(run())


def test_agui_stream_emits_lifecycle_toolcall_and_status():
    events = _collect(
        {"messages": [{"role": "user", "content": "Use run_python to compute 6*7 and report it."}],
         "threadId": "t", "runId": "r"},
        bundles=("code",),
    )
    types = _event_types(events)
    assert "RunStartedEvent" in types and "RunFinishedEvent" in types
    assert "ToolCallStartEvent" in types                       # built-in run_python surfaced
    statuses = [e for e in events if type(e).__name__ == "CustomEvent" and getattr(e, "name", "") == "sheath.status"]
    assert any(s.value.get("tool") == "run_python" for s in statuses)


def test_agui_stream_calls_a_frontend_tool():
    events = _collect(
        {"messages": [{"role": "user", "content": "Render a bar chart of [1,2,3] with show_chart."}],
         "threadId": "t", "runId": "r",
         "tools": [{"name": "show_chart", "description": "Render a bar chart",
                    "parameters": {"type": "object",
                                   "properties": {"values": {"type": "array", "items": {"type": "number"}}},
                                   "required": ["values"]}}]},
        bundles=(),
    )
    names = [getattr(e, "tool_call_name", None) for e in events]
    assert "show_chart" in names                                # request-defined tool was called


def test_agui_stream_overlays_mcp_status():
    from agent_framework import MCPStdioTool

    mcp = MCPStdioTool(name="statusfix", command=sys.executable, args=[_FIXTURE])
    events = _collect(
        {"messages": [{"role": "user", "content": "Call the slow tool with n=3."}],
         "threadId": "t", "runId": "r"},
        tools=[mcp], bundles=(),
    )
    statuses = [e for e in events if type(e).__name__ == "CustomEvent" and getattr(e, "name", "") == "sheath.status"]
    assert any(s.value.get("tool") == "mcp:statusfix" for s in statuses)   # MCP logging overlaid
    assert any(s.value.get("tool") == "slow" and s.value.get("current") is not None for s in statuses)  # MCP progress
