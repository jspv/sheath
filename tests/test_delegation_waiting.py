"""Waiting agents release their concurrency slot and pause their clock (``AgentNode.waiting``)."""

from __future__ import annotations

import asyncio

import pytest

from sheath import SheathConfig, Session
from sheath.delegation import AgentNode, Limits
from sheath.subagents import ConfinedSubagent
from sheath.testing import StubChatClient, text, tool_call
from tests.test_subagents import spec

GUARD_S = 5


async def _build(sess, a_tool, b_tool=None):
    """Declared sub-agent ``a`` (script: call ``a_tool``, then answer "a-done") and, when
    ``b_tool`` is given, sub-agent ``b`` created by ``a`` (call ``b_tool``, answer "b-done")."""
    await sess.create_agent(
        StubChatClient([text("x")]), bundles=("files",),
        subagents={"a": spec([tool_call(a_tool.__name__), text("a-done")], tools=[a_tool])})
    d = sess.delegation
    a = d.node("a")
    if b_tool is not None:
        inner = await sess.create_agent(
            StubChatClient([tool_call(b_tool.__name__), text("b-done")]), tools=[b_tool],
            bundles=("files",), name="b")
        b = AgentNode(name="b", path="a/b", depth=2, creator=a,
                      limits=Limits(timeout_s=sess.config.subagent_task_timeout_s,
                                    max_round_trips=None, subtree_allowance=None,
                                    remaining_depth=0))
        b.agent = ConfinedSubagent(inner, node=b, session=sess)
        d.add_node(b)
    return a


async def _run(node):
    return await node.agent.run("go", session=node.agent.create_session())


def _start_b(a):
    """Start a task on ``b`` as ``a`` (through reach) and return its record."""
    d = a._delegation
    assert d.start_task(a, "b", "go", "b") == "Background task 1 started on agent 'b'."
    return d._owned(a)[-1]


def test_waiting_agent_releases_slot(tmp_path):
    nodes: dict = {}

    async def b_work() -> str:
        """B's work."""
        return "worked"

    async def host_wait() -> str:
        """Start b and wait for it."""
        a = nodes["a"]
        rec = _start_b(a)
        async with a.waiting():
            await asyncio.wait({rec.future})
        nodes["b_done_in_wait"] = rec.future.done()
        return "b finished"

    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, host_wait, b_work)
            resp = await asyncio.wait_for(_run(a), GUARD_S)
            rec = sess.delegation._owned(a)[0]
            await asyncio.wait({rec.future})
            return resp.text, rec.future.result().text, sess._subagent_slots._value

    a_text, b_text, free = asyncio.run(main())
    assert nodes["b_done_in_wait"] is True     # B ran while A waited, not after A finished
    assert a_text == "a-done"
    assert b_text == "b-done"
    assert free == 1


def test_wait_time_not_counted(tmp_path):
    nodes: dict = {}

    async def nap() -> str:
        """Sleep past the bound."""
        await asyncio.sleep(1.0)
        return "slept"

    async def host_wait() -> str:
        """Start b and wait 1.0 s, on b and then some."""
        a = nodes["a"]
        loop = asyncio.get_running_loop()
        began = loop.time()
        rec = _start_b(a)
        async with a.waiting():
            await asyncio.wait({rec.future})
            await asyncio.sleep(max(0.0, 1.0 - (loop.time() - began)))
        nodes["waited"] = loop.time() - began
        return "waited"

    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_task_timeout_s=0.5)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, host_wait, nap)
            resp = await asyncio.wait_for(_run(a), GUARD_S)
            rec = sess.delegation._owned(a)[0]
            await asyncio.wait({rec.future})
            return resp.text, rec.future.exception()

    a_text, b_exc = asyncio.run(main())
    assert nodes["waited"] >= 1.0
    assert a_text == "a-done"
    assert isinstance(b_exc, TimeoutError)
    assert "subagent_task_timeout_s" in str(b_exc)


def test_working_time_still_bounded(tmp_path):
    nodes: dict = {}

    async def work() -> str:
        """Wait briefly (not counted), then work past the bound."""
        async with nodes["a"].waiting():
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)
        return "worked"

    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_task_timeout_s=0.5)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, work)
            await asyncio.wait_for(_run(a), GUARD_S)

    with pytest.raises(TimeoutError, match="subagent_task_timeout_s"):
        asyncio.run(main())


def test_waiting_is_a_no_op_outside_a_run(tmp_path):
    async def unused() -> str:
        """Unused."""
        return "x"

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            a = await _build(sess, unused)
            async with a.waiting():
                pass
            async with sess.delegation.top("top").waiting():
                pass
            return sess._subagent_slots

    assert asyncio.run(main()) is None


def test_cancelled_while_waiting_does_not_leak_slot(tmp_path):
    nodes: dict = {}

    async def hang() -> str:
        """Wait forever, slot released."""
        nodes["entered"].set()
        async with nodes["a"].waiting():
            await asyncio.Event().wait()
        return "never"

    async def main():
        nodes["entered"] = asyncio.Event()
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, hang)
            run = asyncio.create_task(_run(a))
            await asyncio.wait_for(nodes["entered"].wait(), GUARD_S)
            await asyncio.sleep(0.05)
            assert sess._subagent_slots._value == 1      # released while waiting
            run.cancel()
            await asyncio.wait({run}, timeout=GUARD_S)
            assert run.cancelled()
            return sess._subagent_slots._value

    assert asyncio.run(main()) == 1


def test_cancelled_while_reacquiring_does_not_leak_slot(tmp_path):
    """A leaves waiting() while another run holds the only slot, and is cancelled while it
    queues to get it back: it must not release a slot it does not hold."""
    nodes: dict = {}

    async def b_hold() -> str:
        """Hold the slot until told."""
        nodes["b_running"].set()
        await nodes["b_go"].wait()
        return "held"

    async def host() -> str:
        """Start b, wait for it to run, then try to resume while it holds the slot."""
        a = nodes["a"]
        _start_b(a)
        async with a.waiting():
            await nodes["b_running"].wait()
            nodes["a_leaving"].set()
        return "resumed"

    async def main():
        for k in ("b_running", "b_go", "a_leaving"):
            nodes[k] = asyncio.Event()
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, host, b_hold)
            run = asyncio.create_task(_run(a))
            await asyncio.wait_for(nodes["a_leaving"].wait(), GUARD_S)
            await asyncio.sleep(0.05)                    # A now queues for the slot B holds
            assert not run.done()
            run.cancel()
            await asyncio.wait({run}, timeout=GUARD_S)
            assert run.cancelled()
            nodes["b_go"].set()
            rec = sess.delegation._owned(a)[0]
            await asyncio.wait_for(asyncio.wait({rec.future}), GUARD_S)
            assert rec.future.result().text == "b-done"
            return sess._subagent_slots._value

    assert asyncio.run(main()) == 1


def test_wait_first_releases_slot(tmp_path):
    """The wait tool's own wait runs inside waiting(): with one slot, A can wait on B."""
    nodes: dict = {}

    async def b_work() -> str:
        """B's work."""
        return "worked"

    async def host() -> str:
        """Start b and wait for it with the delegation's wait."""
        a = nodes["a"]
        _start_b(a)
        nodes["got"] = await a._delegation.wait_first(a, [1], timeout_s=3)
        return nodes["got"]

    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, host, b_work)
            resp = await asyncio.wait_for(_run(a), GUARD_S)
            return resp.text, nodes["got"], sess._subagent_slots._value

    assert asyncio.run(main()) == ("a-done", "Task 1 finished with status: completed.", 1)


def test_cancelled_wait_does_not_queue_for_slot(tmp_path):
    """Cancelled inside waiting() while another run holds the only slot, A ends at once: its
    teardown does not queue to take back a slot only to release it."""
    nodes: dict = {}

    async def b_hold() -> str:
        """Hold the slot until told."""
        nodes["b_running"].set()
        await nodes["b_go"].wait()
        return "held"

    async def host() -> str:
        """Start b and wait on it forever."""
        a = nodes["a"]
        _start_b(a)
        async with a.waiting():
            await asyncio.Event().wait()
        return "never"

    async def main():
        for k in ("b_running", "b_go"):
            nodes[k] = asyncio.Event()
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=1)
        async with Session.create(cfg) as sess:
            nodes["a"] = a = await _build(sess, host, b_hold)
            run = asyncio.create_task(_run(a))
            await asyncio.wait_for(nodes["b_running"].wait(), GUARD_S)
            run.cancel()
            await asyncio.wait({run}, timeout=1.0)
            ended_while_b_held = run.done()
            nodes["b_go"].set()
            await asyncio.wait({run}, timeout=GUARD_S)
            rec = sess.delegation._owned(a)[0]
            await asyncio.wait_for(asyncio.wait({rec.future}), GUARD_S)
            return ended_while_b_held, run.cancelled(), sess._subagent_slots._value

    assert asyncio.run(main()) == (True, True, 1)


# --- parallel waits in one run (MAF may run a run's tool calls concurrently) --------------------

def _bare_run(slots):
    """A run of a fresh node holding one of ``slots``, set as the current run."""
    from sheath.delegation import CURRENT_RUN, RunState

    node = AgentNode(name="a", path="a", depth=1)
    run = RunState(node=node, slots=slots)
    CURRENT_RUN.set(run)
    return node, run


async def _wait_on(node, ev):
    async with node.waiting():
        await ev.wait()


def test_parallel_wait_entering_during_reacquire_shares_it():
    """Wait 1 leaves and queues to take the slot back; wait 2 enters and leaves meanwhile. They
    share one re-acquire: after teardown the semaphore is back to N, not one short."""

    async def main():
        sem = asyncio.Semaphore(2)
        node, run = _bare_run(sem)
        await run.acquire()
        e1, e2 = asyncio.Event(), asyncio.Event()
        t1 = asyncio.create_task(_wait_on(node, e1))
        await asyncio.sleep(0)
        await sem.acquire()
        await sem.acquire()                      # two other runs fill both slots
        e1.set()
        await asyncio.sleep(0.01)                # t1 queued to re-acquire
        t2 = asyncio.create_task(_wait_on(node, e2))
        await asyncio.sleep(0)
        e2.set()
        await asyncio.sleep(0.01)                # t2 left too
        sem.release()
        sem.release()                            # the other runs end
        await asyncio.wait_for(asyncio.gather(t1, t2), GUARD_S)
        assert run.held
        run.release()                            # the run ends
        await asyncio.sleep(0.01)
        return sem._value

    assert asyncio.run(main()) == 2


def test_parallel_wait_during_reacquire_keeps_clock_paused():
    """A wait entering while another queues to re-acquire does not restart the paused interval:
    a 0.2 s bound is not hit after 1.0 s paused plus a parallel wait."""

    async def main():
        sem = asyncio.Semaphore(1)
        node, run = _bare_run(sem)
        await run.acquire()
        async with asyncio.timeout(0.2) as bound:
            run.bound = bound
            e1, e2 = asyncio.Event(), asyncio.Event()
            t1 = asyncio.create_task(_wait_on(node, e1))
            await asyncio.sleep(0)
            await sem.acquire()                  # another run takes the slot
            await asyncio.sleep(1.0)             # paused far past the bound
            e1.set()
            await asyncio.sleep(0.01)            # t1 queued to re-acquire
            t2 = asyncio.create_task(_wait_on(node, e2))
            await asyncio.sleep(0.01)            # t2 waiting
            sem.release()                        # the other run ends
            await asyncio.sleep(0.01)
            assert not run.held                  # t2 still waits: slot handed back
            e2.set()
            await asyncio.wait_for(asyncio.gather(t1, t2), GUARD_S)
            assert run.held
            remaining = bound.when() - asyncio.get_running_loop().time()
        run.release()
        return remaining, sem._value

    remaining, free = asyncio.run(main())
    assert 0.1 < remaining <= 0.2
    assert free == 1


def test_run_acquire_twice_is_refused():
    async def main():
        _, run = _bare_run(asyncio.Semaphore(2))
        await run.acquire()
        with pytest.raises(RuntimeError):
            await run.acquire()

    asyncio.run(main())


def test_wait_outliving_its_bound_does_not_raise():
    """A wait still open when the run's bound exits leaves it without touching the finished
    Timeout (rescheduling one raises)."""

    async def main():
        node, run = _bare_run(asyncio.Semaphore(1))
        await run.acquire()
        ev = asyncio.Event()
        async with asyncio.timeout(5) as bound:
            run.bound = bound
            t = asyncio.create_task(_wait_on(node, ev))
            await asyncio.sleep(0.01)
        ev.set()
        await asyncio.wait_for(t, GUARD_S)
        run.close()

    asyncio.run(main())
