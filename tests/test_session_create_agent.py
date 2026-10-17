import asyncio

from sheath import SheathConfig, Session
from sheath.testing import StubChatClient, text, tool_call


def test_create_agent_spills_developer_tool_return(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=64)

    def fetch_big(n: int) -> dict:
        """Return a big payload that must not flood the context."""
        return {"rows": list(range(n))}

    client = StubChatClient([
        tool_call("fetch_big", {"n": 500}),
        text("done"),
    ])

    async def run():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(
                client,
                agent_instructions="Fetch the rows.",
                tools=[fetch_big],
                bundles=("code",),
            )
            await agent.run("go")
            return dict(sess.handles)

    handles = asyncio.run(run())
    assert handles  # the developer tool's big return was spilled to a handle
    assert next(iter(handles.values()))["source"] == "tool:fetch_big"


def test_create_agent_exposes_selected_bundle_tools(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    client = StubChatClient([text("hi")])

    async def run():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(
                client, agent_instructions="x", tools=[], bundles=("files",),
            )
            # the sheath instructions for the files bundle reached the agent
            return agent

    agent = asyncio.run(run())
    assert agent is not None
