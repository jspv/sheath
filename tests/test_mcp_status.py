import asyncio

from agent_framework import FunctionTool

from sheath import SheathConfig, Session
from sheath.mcp_status import (
    inject_progress_tokens,
    install_status_wrappers,
)
from sheath.status import StatusBus
from sheath.testing import StubChatClient, text


# --- fakes mimicking MAF's MCPTool surface + mcp notification objects --------

class _Fn:
    def __init__(self, name):
        self.name = name


class _FakeTool:
    def __init__(self, with_seams=True):
        self.name = "srv"
        self.functions = [_Fn("slow")]
        self.log_seen = []
        self.msg_seen = []
        if with_seams:
            self._tool_call_meta_by_name = {}

    async def logging_callback(self, params):
        self.log_seen.append(params)

    async def message_handler(self, message):
        self.msg_seen.append(message)


class _BareTool:
    """No logging_callback / message_handler / _tool_call_meta_by_name."""
    def __init__(self):
        self.name = "bare"
        self.functions = [_Fn("t")]


class _LogParams:
    def __init__(self, data, level="info"):
        self.data = data
        self.level = level


class _ProgressParams:
    def __init__(self, token, progress, total, message):
        self.progressToken = token
        self.progress = progress
        self.total = total
        self.message = message


class _Root:
    def __init__(self, method, params=None):
        self.method = method
        self.params = params


class _Message:
    def __init__(self, root):
        self.root = root


def _bus():
    bus = StatusBus()
    events = []
    bus.subscribe(events.append)
    return bus, events


def test_logging_wrapper_emits_and_chains():
    bus, events = _bus()
    tool = _FakeTool()
    install_status_wrappers(bus, tool, "srv")
    asyncio.run(tool.logging_callback(_LogParams("hello there")))
    assert len(events) == 1
    assert events[0].tool == "mcp:srv"
    assert events[0].message == "hello there"
    assert len(tool.log_seen) == 1                     # original still called


def test_progress_wrapper_emits_with_tool_name_and_chains():
    bus, events = _bus()
    tool = _FakeTool()
    token_map = install_status_wrappers(bus, tool, "srv")
    inject_progress_tokens(tool, "srv", token_map)     # populates token -> "slow"
    token = tool._tool_call_meta_by_name["slow"]["progressToken"]
    msg = _Message(_Root("notifications/progress", _ProgressParams(token, 2, 4, "step 2")))
    asyncio.run(tool.message_handler(msg))
    assert len(events) == 1
    e = events[0]
    assert (e.tool, e.message, e.current, e.total) == ("slow", "step 2", 2, 4)
    assert len(tool.msg_seen) == 1                      # original still called


def test_progress_unknown_token_falls_back_to_server():
    bus, events = _bus()
    tool = _FakeTool()
    install_status_wrappers(bus, tool, "srv")
    msg = _Message(_Root("notifications/progress", _ProgressParams("nope", 1, 2, "x")))
    asyncio.run(tool.message_handler(msg))
    assert events[0].tool == "mcp:srv"


def test_non_progress_message_does_not_emit_but_chains():
    bus, events = _bus()
    tool = _FakeTool()
    install_status_wrappers(bus, tool, "srv")
    msg = _Message(_Root("notifications/tools/list_changed"))
    asyncio.run(tool.message_handler(msg))
    assert events == []
    assert len(tool.msg_seen) == 1                      # original still called


def test_inject_sets_token_in_meta_and_map():
    tool = _FakeTool()
    token_map = {}
    inject_progress_tokens(tool, "srv", token_map)
    assert tool._tool_call_meta_by_name["slow"]["progressToken"] == "sheath:srv:slow"
    assert token_map == {"sheath:srv:slow": "slow"}


def test_inject_preserves_existing_meta():
    tool = _FakeTool()
    tool._tool_call_meta_by_name["slow"] = {"keep": 1}
    inject_progress_tokens(tool, "srv", {})
    assert tool._tool_call_meta_by_name["slow"]["keep"] == 1
    assert "progressToken" in tool._tool_call_meta_by_name["slow"]


def test_graceful_degradation_when_seams_absent():
    bus, events = _bus()
    tool = _BareTool()
    token_map = install_status_wrappers(bus, tool, "bare")   # must not raise
    inject_progress_tokens(tool, "bare", token_map)          # must not raise
    assert token_map == {}
    assert not hasattr(tool, "_tool_call_meta_by_name")
    assert not events


def test_message_handler_passes_through_an_exception():
    # The real mcp message_handler also receives bare Exceptions, not just notifications.
    bus, events = _bus()
    tool = _FakeTool()
    install_status_wrappers(bus, tool, "srv")
    asyncio.run(tool.message_handler(RuntimeError("transport boom")))
    assert events == []
    assert len(tool.msg_seen) == 1                            # original still called


def test_emit_error_does_not_break_the_wrapped_handler():
    bus, _ = _bus()

    def boom(_event):
        raise RuntimeError("emit boom")

    bus.emit = boom                                          # emit itself raises
    tool = _FakeTool()
    install_status_wrappers(bus, tool, "srv")
    asyncio.run(tool.logging_callback(_LogParams("hello")))  # must not raise
    assert len(tool.log_seen) == 1                            # original still called


class _SeamMCPTool:
    """A fake MCPTool that exposes the MAF seams, so _attach_mcp wires status into it."""

    def __init__(self):
        self.name = "fakemcp"
        self.functions: list = []
        self.closed = False
        self._tool_call_meta_by_name: dict = {}
        self.orig_logs: list = []

    async def logging_callback(self, params):
        self.orig_logs.append(params)

    async def message_handler(self, message):
        pass

    async def connect(self):
        def slow(n: int) -> str:
            """A slow MCP tool."""
            return "ok"
        self.functions = [FunctionTool(func=slow, name="slow", description="slow")]

    async def close(self):
        self.closed = True


def test_attach_mcp_installs_wrappers_and_injects_token(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    mcp = _SeamMCPTool()

    async def run():
        events = []
        async with Session.create(cfg) as sess:
            sess.subscribe(events.append)
            await sess.create_agent(
                StubChatClient([text("x")]), agent_instructions="x",
                tools=[mcp], bundles=("code",),
            )
            # token injected for the connected tool
            assert mcp._tool_call_meta_by_name["slow"]["progressToken"] == "sheath:fakemcp:slow"
            # the wrapped logging handler now emits to the bus AND chains the original
            await mcp.logging_callback(type("P", (), {"data": "hi", "level": "info"})())
            return events, mcp

    events, mcp = asyncio.run(run())
    assert any(e.tool == "mcp:fakemcp" and e.message == "hi" for e in events)
    assert mcp.orig_logs                              # original logging_callback still called
    assert mcp.closed
