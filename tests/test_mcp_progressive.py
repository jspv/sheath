"""MCP progressive disclosure: tools the model loads mid-run are wired like the visible ones."""
import sys
from pathlib import Path

import pytest
from agent_framework import MCPStdioTool

from sheath import Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call

# MAF flags the feature itself as experimental; that warning is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:.*MCP progressive disclosure is experimental")

_FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_deferred_server.py")


class _RecordingClient(StubChatClient):
    """Stub that records the text of every message the model is sent."""

    def __init__(self, script):
        super().__init__(script)
        self.messages_seen: list[str] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.messages_seen.append(" ".join(
            str(getattr(c, "result", None) or getattr(c, "text", None) or "")
            for m in messages for c in m.contents))
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def _deferred_mcp() -> MCPStdioTool:
    return MCPStdioTool(name="deferred", command=sys.executable, args=[_FIXTURE],
                        use_progressive_disclosure=True, always_load=["ping"])


def test_a_tool_loaded_mid_run_spills_to_a_handle(tmp_path):
    client = _RecordingClient([
        tool_call("load_tool", {"tool": "big"}, call_id="c1"),
        tool_call("big", {"n": 2000}, call_id="c2"),
        text("done"),
    ])
    h = Sheath(SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=1024),
               client=client, tools=[_deferred_mcp()])
    result = h.solve("go", keep=True)

    assert result.final_text == "done"
    sources = [rec["source"] for rec in result.handles.values()]
    assert "tool:big" in sources
    transcript = client.messages_seen[-1]
    assert "row-01999" not in transcript          # the raw page never reached the model
    assert any(hid in transcript for hid in result.handles)   # its handle summary did


def test_a_tool_loaded_mid_run_reports_progress(tmp_path):
    events = []
    client = StubChatClient([
        tool_call("load_tool", {"tool": "slow"}, call_id="c1"),
        tool_call("slow", {"n": 3}, call_id="c2"),
        text("done"),
    ])
    h = Sheath(SheathConfig(root_dir=tmp_path / "r"), client=client,
               tools=[_deferred_mcp()], on_status=events.append)
    h.solve("go")

    progress = [e for e in events if e.tool == "slow" and e.current is not None]
    assert progress, f"no progress events captured; got {[(e.tool, e.message) for e in events]}"
    assert progress[-1].current == 3 and progress[-1].total == 3


def test_progressive_disclosure_without_the_seam_fails_closed():
    from sheath.session import _every_mcp_function

    class _NoSeam:   # a future MAF that renamed _filtered_functions
        use_progressive_disclosure = True

    with pytest.raises(RuntimeError, match="spill parser"):
        _every_mcp_function(_NoSeam(), [])
