import asyncio

import pytest

from tether import TetherConfig, Tether, tool_factory
from tether.manager import SessionManager
from tether.testing import StubChatClient, text


def _tether(tmp_path):
    return Tether(TetherConfig(root_dir=tmp_path / "base"), client=StubChatClient([text("x")]))


def test_open_is_reuse_or_create_and_isolated_roots(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path))
        a = await m.aopen("t1")
        a2 = await m.aopen("t1")          # same id -> same Conversation
        b = await m.aopen("t2")           # different id -> different Conversation + root
        assert a is a2
        assert b is not a
        assert a.session.root != b.session.root      # isolated per-id roots
        await m.aclose()

    asyncio.run(run())


def test_get_and_close_are_idempotent(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path))
        c = await m.aopen("t1")
        assert m.get("t1") is c
        await m.close("t1")
        assert m.get("t1") is None
        await m.close("t1")               # idempotent: no error

    asyncio.run(run())


def test_lazy_ttl_expiry(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=60)
        c1 = await m.aopen("t1")
        c1.last_activity -= 1000          # simulate idle past the TTL
        assert m.get("t1") is None        # lazy expiry on get
        c2 = await m.aopen("t1")          # open re-creates a fresh conversation
        assert c2 is not c1
        await m.aclose()

    asyncio.run(run())


def test_open_rejects_path_traversal_id(tmp_path):
    # An untrusted threadId must not escape the base via separators/.. — it would be rmtree'd on
    # close. The open fails closed (ValueError) before any workspace is created.
    async def run():
        m = SessionManager(_tether(tmp_path))
        for bad in ("../../../etc/evil", "a/b", "..", r"a\b"):
            with pytest.raises(ValueError):
                await m.aopen(bad)
        assert m.get("../../../etc/evil") is None        # nothing was registered
        await m.aclose()

    asyncio.run(run())


def test_sweep_reaps_idle(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=60)
        c = await m.aopen("t1")
        c.last_activity -= 1000
        await m.sweep()
        assert m.get("t1") is None

    asyncio.run(run())


class _FakeMCP:
    """Stands in for a connected MCP server: identity matters, and so does close()."""
    def __init__(self) -> None:
        self.closed = False
        self.functions: list = []         # lets Session recognise this as an MCP server

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


def test_each_conversation_gets_its_own_tool_instance(tmp_path):
    built = []

    def build():
        instance = _FakeMCP()
        built.append(instance)
        return instance

    async def run():
        tether = Tether(TetherConfig(root_dir=tmp_path / "base"),
                        client=StubChatClient([text("x")]),
                        tools=[tool_factory(build)])
        try:
            a = await tether.aopen("conv-a")
            b = await tether.aopen("conv-b")
            assert a is not b
            assert len(built) == 2
            assert built[0] is not built[1]   # no shared connection across conversations

            await tether._sessions().close("conv-a")
            assert built[0].closed is True
            assert built[1].closed is False   # closing one must not close the other's
        finally:
            await tether.aclose_sessions()

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "x\x00y", "a\nb", "x" * 300])
def test_conversation_ids_must_be_a_single_safe_segment(tmp_path, bad):
    async def run():
        m = SessionManager(_tether(tmp_path))
        with pytest.raises(ValueError):
            await m.aopen(bad)

    asyncio.run(run())


def test_busy_conversation_is_never_expired(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=0.01)
        conv = await m.aopen("t1")
        async with conv._lock:                    # a turn is in flight
            await asyncio.sleep(0.05)             # well past the TTL
            await m.sweep()
            assert m.get("t1") is conv and conv.session.root.exists()
            assert await m.aopen("t1") is conv
        await m.aclose()

    asyncio.run(run())


def test_reopening_a_live_conversation_counts_as_activity(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=60)
        conv = await m.aopen("t1")
        conv.last_activity -= 30
        before = conv.last_activity
        assert await m.aopen("t1") is conv
        assert conv.last_activity > before
        await m.aclose()

    asyncio.run(run())


def test_agui_stream_keeps_an_active_thread_alive_past_its_ttl(tmp_path, monkeypatch):
    async def run():
        h = Tether(TetherConfig(root_dir=tmp_path / "base", idle_ttl_s=0.5),
                   client=StubChatClient([text("x")]))

        async def fake(agent, bus, input_data, **kwargs):
            yield "event"

        monkeypatch.setattr("tether.agui.agui_event_stream", fake)
        convs = []
        for turn in range(4):                      # 4 turns, 0.2 s apart: 0.6 s > TTL overall
            async for _ in h.agui_stream({"threadId": "t1", "messages": []}):
                pass
            conv = h._sessions().get("t1")
            convs.append(conv)
            if turn == 0:
                (conv.session.root / "kept.txt").write_text("from turn 0")
            await asyncio.sleep(0.2)
        assert all(c is convs[0] for c in convs)   # never reaped and recreated
        assert (convs[0].session.root / "kept.txt").read_text() == "from turn 0"
        await h.aclose_sessions()

    asyncio.run(run())
