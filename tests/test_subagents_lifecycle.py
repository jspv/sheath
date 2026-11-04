"""Sub-agent lifecycle: no task outlives its turn, its MCP servers or its workspace (issue #9)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import aclosing

import pytest
from agent_framework import BackgroundAgentsProvider

from sheath import Conversation, SheathConfig, Session
from sheath.testing import StubChatClient, text, tool_call
from tests.test_subagents import spec, start


def stop(**kw):
    """loop_should_continue that never re-runs the parent: it ends with the task running."""
    return False


class Gate:
    """The sub's tool sets ``entered`` on entry; the parent's ``gate`` tool waits for it, so
    the parent turn ends only once the sub-agent is inside its tool."""

    def __init__(self):
        self.entered = threading.Event()

    def tool(self):
        entered = self.entered

        async def gate() -> str:
            """Wait for the sub-agent to start."""
            ok = await asyncio.to_thread(entered.wait, 5)
            return "in" if ok else "timeout"
        return gate


def sleeper(gate, flag, marker):
    async def nap() -> str:
        """Sleep, then write a marker."""
        gate.entered.set()
        try:
            await asyncio.sleep(5)
            marker.write_text("late")
        finally:
            flag["cancelled"] = True
        return "woke"
    return nap


def parent_script():
    return [start("a", 1), tool_call("gate", call_id="g1"), text("bye"), text("bye"),
            text("bye")]


def test_session_close_cancels_running_subagent(tmp_path):
    gate, flag = Gate(), {}
    marker = tmp_path / "marker.txt"
    subs = {"a": spec([tool_call("nap"), text("never")], tools=[sleeper(gate, flag, marker)])}

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(parent_script()),
                                            tools=[gate.tool()], bundles=("files",),
                                            subagents=subs, loop_should_continue=stop)
            await agent.run("go", session=agent.create_session())
            assert sess.subagents_running == 1
            t0 = time.monotonic()
        # Checked here, before asyncio.run's own shutdown would cancel leftover tasks.
        return time.monotonic() - t0, flag.get("cancelled"), sess.subagents_running

    elapsed, cancelled, running = asyncio.run(main())
    assert cancelled is True
    assert running == 0
    assert not marker.exists()
    assert elapsed < 2.5


def test_conversation_close_releases_and_reaps_after_cancel(tmp_path, monkeypatch):
    gate, flag = Gate(), {}
    marker = tmp_path / "marker.txt"
    released: list = []
    real = BackgroundAgentsProvider.release_session

    async def spy(self, session, **kw):
        released.append(session)
        return await real(self, session, **kw)

    monkeypatch.setattr(BackgroundAgentsProvider, "release_session", spy)
    subs = {"a": spec([tool_call("nap"), text("never")], tools=[sleeper(gate, flag, marker)])}

    async def main():
        conv = await Conversation.acreate(
            id="c1", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient(parent_script()), tools=[gate.tool()], bundles=("files",),
            agent_kwargs={"subagents": subs, "loop_should_continue": stop})
        await conv.aask("go")
        await conv.aclose()
        await asyncio.sleep(0.2)
        return conv

    conv = asyncio.run(main())
    assert released and all(s is conv.agent_session for s in released)
    assert not (tmp_path / "r").exists()
    assert flag.get("cancelled") is True
    assert not marker.exists()


def test_reap_waits_for_in_flight_sync_file_tool(tmp_path):
    gate, state = Gate(), {}
    holder: dict = {}

    def hold() -> str:
        """Hold the workspace lock, then write a file."""
        sess = holder["sess"]
        with sess.io_lock:
            gate.entered.set()
            time.sleep(0.3)
            (sess.root / "late.txt").write_text("x")
            state["finished"] = True
        return "held"

    subs = {"a": spec([tool_call("hold"), text("never")], tools=[hold])}

    async def main():
        conv = await Conversation.acreate(
            id="c1", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient(parent_script()), tools=[gate.tool()], bundles=("files",),
            agent_kwargs={"subagents": subs, "loop_should_continue": stop})
        holder["sess"] = conv.session
        await conv.aask("go")
        await conv.aclose()
        return state.get("finished")

    finished_at_close = asyncio.run(main())
    assert finished_at_close is True
    assert not (tmp_path / "r").exists()


def test_turn_end_cancels_leftover_tasks(tmp_path):
    gate, flag = Gate(), {}
    marker = tmp_path / "marker.txt"
    subs = {"a": spec([tool_call("nap"), text("never")], tools=[sleeper(gate, flag, marker)])}

    async def main():
        conv = await Conversation.acreate(
            id="c1", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient(parent_script()), tools=[gate.tool()], bundles=("files",),
            agent_kwargs={"subagents": subs, "loop_max_iterations": 1})
        t0 = time.monotonic()
        await conv.aask("go")
        elapsed = time.monotonic() - t0
        running = conv.session.subagents_running
        cancelled = flag.get("cancelled")
        await conv.aclose()
        return elapsed, running, cancelled

    elapsed, running, cancelled = asyncio.run(main())
    assert elapsed < 2.5
    assert cancelled is True
    assert running == 0
    assert not marker.exists()


def test_agui_disconnect_cancels_tasks(tmp_path):
    gate, flag = Gate(), {}
    marker = tmp_path / "marker.txt"
    subs = {"a": spec([tool_call("nap"), text("never")], tools=[sleeper(gate, flag, marker)])}

    async def main():
        conv = await Conversation.acreate(
            id="c1", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient(parent_script()), tools=[gate.tool()], bundles=("files",),
            agent_kwargs={"subagents": subs})
        async with aclosing(conv.astream_agui(
                {"messages": [{"role": "user", "content": "go"}], "threadId": "t1"})) as events:
            async for _ in events:
                if gate.entered.is_set():
                    break
        state = (flag.get("cancelled"), conv.session.subagents_running)
        await conv.aclose()
        return state

    cancelled, running = asyncio.run(main())
    assert cancelled is True
    assert running == 0
    assert not marker.exists()


def test_cleanup_root_uses_locked_removal(tmp_path, monkeypatch):
    import sheath.session as session_mod

    calls: list = []
    real = session_mod._remove_root_locked

    def spy(root, lock):
        calls.append((root, lock, threading.current_thread() is threading.main_thread()))
        return real(root, lock)

    monkeypatch.setattr(session_mod, "_remove_root_locked", spy)

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r", cleanup=True)) as sess:
            (sess.root / "f.txt").write_text("x")
            return sess

    sess = asyncio.run(main())
    assert not (tmp_path / "r").exists()
    assert calls == [(sess.root, sess.io_lock, False)]   # run in a worker thread


def test_subagents_running_counts_queued_task(tmp_path):
    seen: dict = {}
    gate = Gate()
    release = threading.Event()

    async def work() -> str:
        """Work until released."""
        gate.entered.set()
        await asyncio.to_thread(release.wait, 5)
        return "ok"

    holder: dict = {}

    async def probe() -> str:
        """Record how many sub-agent runs are registered."""
        await asyncio.to_thread(gate.entered.wait, 5)
        await asyncio.sleep(0.05)   # let the second task reach its slot wait
        seen["running"] = holder["sess"].subagents_running
        release.set()
        return "probed"

    subs = {"a": spec([tool_call("work"), text("ra")], tools=[work]),
            "b": spec([tool_call("work"), text("rb")], tools=[work])}
    script = [start("a", 1), start("b", 2), tool_call("probe", call_id="p1"),
              tool_call("background_agents_wait_for_first_completion",
                        {"task_ids": [1, 2]}, "w1"),
              tool_call("background_agents_wait_for_first_completion",
                        {"task_ids": [1, 2]}, "w2"),
              text("ok")]

    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            holder["sess"] = sess
            agent = await sess.create_agent(StubChatClient(script), tools=[probe],
                                            bundles=("files",), subagents=subs,
                                            loop_should_continue=stop)
            await agent.run("go", session=agent.create_session())

    asyncio.run(main())
    assert seen["running"] == 2


def _lock_out(root):
    """Ten chmod-000 directories (each holding a file) among ten deletable siblings.

    rmtree's walk order follows the directory listing, which is hash ordered on common file
    systems; with the two interleaved, a walk that stops at its first error leaves siblings.
    """
    locked, siblings = [], []
    for i in range(10):
        for kind, into in (("locked", locked), ("sibling", siblings)):
            d = root / f"{kind}{i}"
            d.mkdir()
            (d / "f").write_text("x")
            into.append(d)
    for d in locked:
        d.chmod(0)
    return locked, siblings


def _unlock(dirs):
    for d in dirs or ():
        if d.exists():
            d.chmod(0o755)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores directory permissions")
def test_conversation_reap_continues_past_unremovable_entry(tmp_path, caplog):
    holder: dict = {}

    async def main():
        conv = await Conversation.acreate(
            id="c1", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient([text("ok")]), bundles=("files",))
        holder["locked"], holder["siblings"] = _lock_out(conv.session.root)
        await conv.aclose()          # must not raise

    try:
        with caplog.at_level(logging.WARNING, logger="sheath"):
            asyncio.run(main())
        assert not [p for p in holder["siblings"] if p.exists()]
        assert all(d.exists() for d in holder["locked"])
        assert any("could not" in r.getMessage() for r in caplog.records)
    finally:
        _unlock(holder.get("locked"))


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores directory permissions")
def test_session_cleanup_continues_past_unremovable_entry(tmp_path, caplog):
    holder: dict = {}

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r", cleanup=True)) as sess:
            holder["locked"], holder["siblings"] = _lock_out(sess.root)

    try:
        with caplog.at_level(logging.WARNING, logger="sheath"):
            asyncio.run(main())   # a partial reap is logged, not raised
        assert not [p for p in holder["siblings"] if p.exists()]
        assert all(d.exists() for d in holder["locked"])
        assert any("could not" in r.getMessage() for r in caplog.records)
    finally:
        _unlock(holder.get("locked"))


def test_session_close_cancels_task_not_yet_started(tmp_path):
    """A task MAF's start tool created but that has not taken a step yet is cancelled too."""
    from agent_framework import SessionContext

    ran: dict = {}

    async def touch() -> str:
        """Record that the sub-agent ran."""
        ran["tool"] = True
        return "ran"

    subs = {"a": spec([tool_call("touch"), text("never")], tools=[touch])}

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient([text("bye")]), bundles=("files",),
                                            subagents=subs)
            provider = sess._subagent_providers[0]
            agent_session = agent.create_session()
            ctx = SessionContext(input_messages=[])
            await provider.before_run(agent=agent, session=agent_session, context=ctx,
                                      state={})
            start_tool = next(t for t in ctx.tools
                              if t.name == "background_agents_start_task")
            # Sync: creates the task; nothing below suspends before the close starts.
            start_tool.func(agent_name="a", input="go", description="a")
            registered = sess.subagents_running
        await asyncio.sleep(0.2)   # a task that escaped cancellation would run its tool here
        return registered

    registered = asyncio.run(main())
    assert registered == 0          # not yet registered when the close began
    assert "tool" not in ran
