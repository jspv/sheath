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
