import asyncio

from tether import TetherConfig
from tether.conversation import Conversation
from tether.testing import StubChatClient, text, tool_call


def _save(hid, value):
    return tool_call("run_python", {"code": f"from tether_sandbox import save\nsave({hid!r}, {value!r})\n"})


def test_conversation_persists_workspace_across_turns(tmp_path):
    # StubChatClient consumes its script linearly across run() calls, so 4 steps = 2 turns.
    client = StubChatClient([
        _save("h1", {"x": 1}), text("saved"),
        tool_call("run_python", {"code": "from tether_sandbox import load, save\nsave('h2', load('h1'))\n"}),
        text("done"),
    ])

    async def run():
        conv = await Conversation.acreate(id="c1", config=TetherConfig(root_dir=tmp_path / "r"),
                                          client=client, tools=[], bundles=("code",))
        r1 = await conv.aask("save it")
        r2 = await conv.aask("reload it")            # turn 2 reuses the same workspace
        loaded_ok = "h2" in r2.handles and conv.session.store.get("h2") == {"x": 1}
        await conv.aclose()
        return r1, r2, loaded_ok, conv

    r1, r2, loaded_ok, conv = asyncio.run(run())
    assert "h1" in r1.handles
    assert loaded_ok                                 # turn 2 read turn-1's handle from the persistent store
    assert not (tmp_path / "r").exists()             # reaped on close (reap_on_close default True)


def test_conversation_keep_on_close(tmp_path):
    client = StubChatClient([text("hi")])

    async def run():
        conv = await Conversation.acreate(id="c2", config=TetherConfig(root_dir=tmp_path / "k"),
                                          client=client, tools=[], bundles=(), reap_on_close=False)
        await conv.aask("hello")
        root = conv.session.root
        await conv.aclose()
        return root

    root = asyncio.run(run())
    assert root.exists()                             # retained when reap_on_close=False


def test_conversation_aclose_is_idempotent(tmp_path):
    client = StubChatClient([text("hi")])

    async def run():
        conv = await Conversation.acreate(id="c4", config=TetherConfig(root_dir=tmp_path / "i"),
                                          client=client, tools=[], bundles=())
        await conv.aask("hello")
        await conv.aclose()
        await conv.aclose()                          # second close must be a no-op, not raise

    asyncio.run(run())                               # completes without error


def test_conversation_error_is_non_fatal(tmp_path):
    class _Boom(StubChatClient):
        async def _inner_get_response(self, *, messages, stream, options, **kwargs):
            raise RuntimeError("model boom")

    async def run():
        conv = await Conversation.acreate(id="c3", config=TetherConfig(root_dir=tmp_path / "e"),
                                          client=_Boom([text("x")]), tools=[], bundles=())
        r = await conv.aask("go")                    # error captured, conversation survives
        ok = r.error is not None and "RuntimeError" in r.error
        await conv.aclose()
        return ok

    assert asyncio.run(run())


def _conv(tmp_path):
    async def make():
        return await Conversation.acreate(
            id="c1", config=TetherConfig(root_dir=tmp_path / "c1"),
            client=StubChatClient([text("ok")]), bundles=("files",))
    return make


def _fake_stream(events, gate=None):
    async def fake(agent, bus, input_data, **kwargs):
        for e in events:
            if gate is not None:
                await gate.wait()
            yield e
    return fake


def test_agui_turn_holds_the_lock_and_stamps_activity(tmp_path, monkeypatch):
    async def run():
        conv = await _conv(tmp_path)()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["a", "b"]))
        conv.last_activity = 0.0
        seen, busy = [], []
        async for event in conv.astream_agui({"messages": []}):
            seen.append(event)
            busy.append(conv.busy)
        assert seen == ["a", "b"] and busy == [True, True]
        assert not conv.busy and conv.last_activity > 0.0
        await conv.aclose()

    asyncio.run(run())


def test_agui_turns_on_one_conversation_are_serialized(tmp_path, monkeypatch):
    async def run():
        conv = await _conv(tmp_path)()
        gate = asyncio.Event()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["x"], gate))
        order = []

        async def turn(label):
            async for _ in conv.astream_agui({"messages": []}):
                order.append(label)

        first = asyncio.create_task(turn("first"))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(turn("second"))
        await asyncio.sleep(0.05)
        assert conv.busy and order == []
        gate.set()
        await asyncio.gather(first, second)
        assert order == ["first", "second"]
        await conv.aclose()

    asyncio.run(run())


def test_abandoned_agui_stream_releases_the_lock(tmp_path, monkeypatch):
    async def run():
        conv = await _conv(tmp_path)()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["a", "b", "c"]))
        stream = conv.astream_agui({"messages": []})
        assert await stream.__anext__() == "a"
        assert conv.busy
        await stream.aclose()                     # what a server does on client disconnect
        assert not conv.busy
        assert (await conv.aask("still usable?")).error is None
        await conv.aclose()

    asyncio.run(run())


def test_closing_agui_stream_tears_down_inner_stream_while_lock_held(tmp_path, monkeypatch):
    async def run():
        conv = await _conv(tmp_path)()
        seen = []

        async def fake(agent, bus, input_data, **kwargs):
            try:
                yield "a"
                yield "b"
            finally:
                seen.append(conv.busy)

        monkeypatch.setattr("tether.agui.agui_event_stream", fake)
        stream = conv.astream_agui({"messages": []})
        assert await stream.__anext__() == "a"
        await stream.aclose()
        assert seen == [True]
        assert not conv.busy
        await conv.aclose()

    asyncio.run(run())


def test_closing_agui_stream_closes_agent_stream_and_status_merge_under_lock(tmp_path, monkeypatch):
    import pytest
    agui_pkg = pytest.importorskip("agent_framework_ag_ui")

    async def run():
        conv = await _conv(tmp_path)()
        seen = []

        class FakeAFA:
            def __init__(self, agent, **kwargs):
                pass

            async def run(self, input_data):
                try:
                    yield "a"
                    yield "b"
                finally:
                    seen.append((conv.busy, len(conv.session.status_bus._subscribers)))

        monkeypatch.setattr(agui_pkg, "AgentFrameworkAgent", FakeAFA)
        stream = conv.astream_agui({"messages": []})
        assert await stream.__anext__() == "a"
        assert len(conv.session.status_bus._subscribers) == 1
        await stream.aclose()
        assert seen == [(True, 1)]               # agent stream closed while lock held
        assert conv.session.status_bus._subscribers == []
        assert not conv.busy
        await conv.aclose()

    asyncio.run(run())


class _GatedAgent:
    """Stands in for the MAF agent: a turn that blocks until ``gate`` is set."""

    def __init__(self, root):
        self.root = root
        self.gate = asyncio.Event()
        self.started = asyncio.Event()
        self.root_seen_during_turn = None

    async def run(self, question, session=None):
        self.started.set()
        await self.gate.wait()
        self.root_seen_during_turn = self.root.exists()

        class _R:
            text = "ok"
        return _R()


def test_aclose_waits_for_a_running_turn(tmp_path):
    async def run():
        conv = await _conv(tmp_path)()
        root = conv.session.root
        agent = conv.agent = _GatedAgent(root)
        turn = asyncio.create_task(conv.aask("go"))
        await agent.started.wait()
        closing = asyncio.create_task(conv.aclose())
        await asyncio.sleep(0.05)
        assert root.exists() and not closing.done()   # teardown waits for the turn
        agent.gate.set()
        result = await turn
        await closing
        return result, agent, root

    result, agent, root = asyncio.run(run())
    assert result.error is None and agent.root_seen_during_turn is True
    assert not root.exists()                         # reaped once the turn finished


def test_turn_queued_behind_aclose_sees_closed(tmp_path):
    async def run():
        conv = await _conv(tmp_path)()
        agent = conv.agent = _GatedAgent(conv.session.root)
        first = asyncio.create_task(conv.aask("go"))
        await agent.started.wait()
        closing = asyncio.create_task(conv.aclose())
        await asyncio.sleep(0.01)
        queued = asyncio.create_task(conv.aask("after close"))
        await asyncio.sleep(0.01)
        agent.gate.set()
        await first
        await closing
        return await queued

    late = asyncio.run(run())
    assert late.error is not None and "closed" in late.error


def test_agui_turn_survives_concurrent_aclose_and_later_turns_refuse(tmp_path, monkeypatch):
    import pytest

    async def run():
        conv = await _conv(tmp_path)()
        root = conv.session.root
        gate = asyncio.Event()
        seen_root = []

        async def fake(agent, bus, input_data, **kwargs):
            yield "a"
            await gate.wait()
            seen_root.append(root.exists())
            yield "b"

        monkeypatch.setattr("tether.agui.agui_event_stream", fake)
        events = []

        async def consume():
            async for e in conv.astream_agui({"messages": []}):
                events.append(e)

        stream = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        closing = asyncio.create_task(conv.aclose())
        await asyncio.sleep(0.05)
        assert root.exists() and not closing.done()
        gate.set()
        await stream
        await closing
        assert events == ["a", "b"] and seen_root == [True] and not root.exists()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in conv.astream_agui({"messages": []}):
                pass

    asyncio.run(run())


def test_aclose_wait_is_bounded(tmp_path, monkeypatch):
    import tether.conversation as conversation_mod

    monkeypatch.setattr(conversation_mod, "_CLOSE_WAIT_S", 0.05)

    async def run():
        conv = await _conv(tmp_path)()
        agent = conv.agent = _GatedAgent(conv.session.root)
        turn = asyncio.create_task(conv.aask("go"))
        await agent.started.wait()
        await asyncio.wait_for(conv.aclose(), timeout=5)   # proceeds despite the stuck turn
        assert not conv.session.root.exists()
        agent.gate.set()
        await turn

    asyncio.run(run())
