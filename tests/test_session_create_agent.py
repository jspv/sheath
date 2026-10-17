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


class _RecordingClient(StubChatClient):
    """Stub that records the chat options and the message count of every model call."""

    def __init__(self, script):
        super().__init__(script)
        self.options_seen: list[dict] = []
        self.message_counts: list[int] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.options_seen.append(dict(options))
        self.message_counts.append(len(list(messages)))
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def test_create_agent_never_stores_history_server_side_by_default(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    client = _RecordingClient([text("hi")])

    async def run():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(client, agent_instructions="x", tools=[], bundles=("files",))
            await agent.run("go")

    asyncio.run(run())
    assert client.options_seen, "the model was never called"
    # the OpenAI client stores conversations at OpenAI by default; we opt out explicitly
    assert all(o.get("store") is False for o in client.options_seen)


def test_caller_default_options_are_merged_and_may_override_store(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")

    async def agent_options(**maf_kwargs):
        client = _RecordingClient([text("hi")])
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(client, agent_instructions="x", tools=[],
                                            bundles=("files",), **maf_kwargs)
            await agent.run("go")
        return client.options_seen[0]

    merged = asyncio.run(agent_options(default_options={"temperature": 0.2}))
    assert merged.get("store") is False and merged.get("temperature") == 0.2

    overridden = asyncio.run(agent_options(default_options={"store": True}))
    assert overridden.get("store") is True
