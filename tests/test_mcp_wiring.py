import asyncio

import pytest
from agent_framework import FunctionTool
from sheath import SheathConfig, Session
from sheath.testing import StubChatClient, text, tool_call


class FakeMCPTool:
    """Mimics MAF's MCPTool surface: connect() populates .functions; close() tears down."""

    def __init__(self):
        self.functions: list[FunctionTool] = []
        self.connected = False
        self.closed = False

    async def connect(self):
        self.connected = True

        def mcp_query(q: str) -> dict:
            """Return a big result from the MCP server."""
            return {"hits": list(range(500)), "q": q}

        self.functions = [FunctionTool(func=mcp_query, name="mcp_query",
                                       description="mcp query")]

    async def close(self):
        self.closed = True


def test_create_agent_connects_and_spills_mcp(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=64)
    mcp = FakeMCPTool()
    client = StubChatClient([
        tool_call("mcp_query", {"q": "widgets"}),
        text("done"),
    ])

    async def run():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(
                client, agent_instructions="query it", tools=[mcp], bundles=("code",),
            )
            assert mcp.connected
            await agent.run("go")
            return dict(sess.handles), mcp

    handles, mcp = asyncio.run(run())
    assert handles  # MCP result over threshold was spilled to a handle
    assert next(iter(handles.values()))["source"] == "tool:mcp_query"
    assert mcp.closed  # connection torn down on context exit


def test_mcp_closed_even_on_error(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    mcp = FakeMCPTool()

    async def run():
        try:
            async with Session.create(cfg) as sess:
                await sess.create_agent(
                    StubChatClient([text("x")]),
                    agent_instructions="x", tools=[mcp], bundles=("code",),
                )
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        return mcp

    mcp = asyncio.run(run())
    assert mcp.connected and mcp.closed  # __aexit__ closed it despite the error


class _BadMCPTool:
    """An MCP server that fails to connect."""
    functions: list = []

    def __init__(self):
        self.closed = False

    async def connect(self):
        raise ConnectionError("server unreachable")

    async def close(self):
        self.closed = True


def test_mcp_connect_failure_names_server_and_closes_prior(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    good = FakeMCPTool()
    bad = _BadMCPTool()

    async def run():
        with pytest.raises(RuntimeError, match="failed to connect MCP server"):
            async with Session.create(cfg) as sess:
                # good connects first (registered), then bad fails -> error propagates
                await sess.create_agent(
                    StubChatClient([text("x")]),
                    agent_instructions="x", tools=[good, bad], bundles=("code",),
                )
        return good, bad

    good, bad = asyncio.run(run())
    assert good.connected and good.closed   # prior server torn down on context exit
    assert not bad.closed                   # bad never connected, so never closed
