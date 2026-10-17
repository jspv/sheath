import asyncio
import importlib.util
import sys
import threading

import pytest

from sheath.agui import merge_status, status_to_agui
from sheath.status import StatusBus, StatusEvent

_HAS_AGUI = importlib.util.find_spec("ag_ui") is not None


def _tag(status_event):
    # a fake to_event mapper so the overlay is testable without ag_ui
    return ("status", status_event.message)


def test_merge_status_yields_source_events_in_order():
    bus = StatusBus()

    async def source():
        yield "A"
        yield "B"

    out = []

    async def run():
        async for ev in merge_status(source(), bus, to_event=_tag):
            out.append(ev)

    asyncio.run(run())
    assert [e for e in out if isinstance(e, str)] == ["A", "B"]


def test_merge_status_interleaves_bus_events():
    bus = StatusBus()

    async def source():
        bus.emit(StatusEvent(tool="t", message="mid-1"))
        yield "A"
        await asyncio.sleep(0)                      # let the scheduled put run
        bus.emit(StatusEvent(tool="t", message="mid-2"))
        yield "B"
        await asyncio.sleep(0)

    out = []

    async def run():
        async for ev in merge_status(source(), bus, to_event=_tag):
            out.append(ev)

    asyncio.run(run())
    statuses = [e for e in out if isinstance(e, tuple)]
    assert ("status", "mid-1") in statuses
    assert ("status", "mid-2") in statuses


def test_merge_status_handles_emit_from_worker_thread():
    bus = StatusBus()

    async def source():
        t = threading.Thread(target=lambda: bus.emit(StatusEvent(tool="bg", message="from-thread")))
        t.start()
        t.join()
        yield "X"
        await asyncio.sleep(0)

    out = []

    async def run():
        async for ev in merge_status(source(), bus, to_event=_tag):
            out.append(ev)

    asyncio.run(run())
    assert ("status", "from-thread") in out          # cross-thread marshaling worked


def test_merge_status_unsubscribes_on_exit():
    bus = StatusBus()

    async def source():
        yield "A"

    async def run():
        async for _ in merge_status(source(), bus, to_event=_tag):
            pass

    asyncio.run(run())
    assert bus._subscribers == []                    # sink removed in finally


def test_merge_status_emitted_during_final_event_is_not_lost():
    bus = StatusBus()

    async def source():
        yield "A"
        bus.emit(StatusEvent(tool="t", message="final"))   # emitted during the LAST event, no trailing await

    out = []

    async def run():
        async for ev in merge_status(source(), bus, to_event=_tag):
            out.append(ev)

    asyncio.run(run())
    assert ("status", "final") in out


@pytest.mark.skipif(not _HAS_AGUI, reason="needs the agui extra (ag-ui-protocol)")
def test_status_to_agui_maps_fields():
    ev = status_to_agui(StatusEvent(tool="run_python", message="running", current=1, total=3))
    assert ev.name == "sheath.status"
    assert ev.value == {"tool": "run_python", "message": "running", "current": 1, "total": 3}


@pytest.mark.skipif(not _HAS_AGUI, reason="needs the agui extra (ag-ui-protocol)")
def test_agui_stream_reuses_workspace_per_thread(tmp_path):
    import asyncio

    from sheath import Sheath, SheathConfig
    from sheath.testing import StubChatClient, text, tool_call

    client = StubChatClient([
        tool_call("run_python", {"code": "from sheath_sandbox import save\nsave('h1', {'x': 1})\n"}),
        text("saved"),
        tool_call("run_python", {"code": "from sheath_sandbox import load, save\nsave('h2', load('h1'))\n"}),
        text("done"),
    ])
    h = Sheath(SheathConfig(root_dir=tmp_path / "base"), client=client, bundles=("code",))

    async def run():
        async for _ in h.agui_stream({"messages": [{"role": "user", "content": "save"}], "threadId": "t1"}):
            pass
        async for _ in h.agui_stream({"messages": [{"role": "user", "content": "reload"}], "threadId": "t1"}):
            pass
        conv = h._sessions().get("t1")               # same workspace reused across both calls
        ok = conv is not None and "h1" in conv.session.handles and conv.session.store.get("h2") == {"x": 1}
        await h.aclose_sessions()
        return ok

    assert asyncio.run(run())


def test_repair_reorders_results_after_their_tool_call():
    # CopilotKit replays tool RESULTS before the assistant message that declared the calls;
    # OpenAI requires the reverse. repair_tool_message_order must rebuild a valid order.
    from sheath.agui import repair_tool_message_order

    msgs = [
        {"role": "user", "content": "q"},
        {"role": "tool", "toolCallId": "c1", "content": "r1"},
        {"role": "tool", "toolCallId": "c2", "content": "r2"},
        {"role": "assistant", "toolCalls": [{"id": "c1"}, {"id": "c2"}], "content": None},
        {"role": "assistant", "content": "the answer"},
        {"role": "user", "content": "q2"},
    ]
    out = repair_tool_message_order(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert out[1].get("toolCalls")                      # tool-call assistant comes first
    assert out[2]["toolCallId"] == "c1" and out[3]["toolCallId"] == "c2"  # then its results, in order


def test_repair_drops_orphan_call_but_keeps_final_answer():
    from sheath.agui import repair_tool_message_order

    msgs = [
        {"role": "assistant", "toolCalls": [{"id": "c1"}], "content": None},  # no matching result
        {"role": "assistant", "content": "final"},
    ]
    out = repair_tool_message_order(msgs)
    assert [m["role"] for m in out] == ["assistant"]    # orphan-only assistant dropped
    assert out[0]["content"] == "final"


def test_repair_leaves_valid_order_unchanged():
    from sheath.agui import repair_tool_message_order

    msgs = [
        {"role": "assistant", "toolCalls": [{"id": "c1"}], "content": None},
        {"role": "tool", "toolCallId": "c1", "content": "r1"},
        {"role": "assistant", "content": "done"},
    ]
    out = repair_tool_message_order(msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "assistant"]
    assert out[0]["toolCalls"][0]["id"] == "c1"


def test_agui_event_stream_errors_clearly_without_the_extra(monkeypatch):
    # Simulate the extra being absent even though it's installed in this env.
    monkeypatch.setitem(sys.modules, "agent_framework_ag_ui", None)
    from sheath.agui import agui_event_stream
    from sheath.status import StatusBus

    async def run():
        gen = agui_event_stream(object(), StatusBus(), {"messages": []})
        await gen.__anext__()

    with pytest.raises(RuntimeError, match="install the 'agui' extra"):
        asyncio.run(run())
