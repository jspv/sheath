import sys
from pathlib import Path

from agent_framework import MCPStdioTool

from sheath import Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call

_FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_progress_server.py")


def test_mcp_logging_and_progress_reach_on_status(tmp_path):
    events = []
    mcp = MCPStdioTool(name="statusfix", command=sys.executable, args=[_FIXTURE])
    client = StubChatClient([tool_call("slow", {"n": 3}), text("done")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "r"), client=client,
                tools=[mcp], on_status=events.append)
    result = h.solve("go")

    assert result.final_text == "done"
    # logging notifications -> attributed to the server
    logs = [e for e in events if e.tool == "mcp:statusfix"]
    assert any("processing item" in e.message for e in logs)
    # progress notifications -> attributed to the emitting tool, with current/total
    progress = [e for e in events if e.tool == "slow" and e.current is not None]
    assert progress, f"no progress events captured; got {[(e.tool, e.message) for e in events]}"
    assert progress[-1].current == 3 and progress[-1].total == 3
