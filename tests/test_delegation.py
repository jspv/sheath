"""sheath's own delegation provider: parity with MAF's background-agents tools."""

from __future__ import annotations

import asyncio

from agent_framework import BackgroundAgentsProvider

from sheath import SheathConfig, Session
from sheath.delegation import (
    AgentNode,
    DelegationProvider,
    tasks_running,
    tasks_running_message,
)
from sheath.testing import StubChatClient, text, tool_call
from tests.test_subagents import cont, record, results, spec, start, wait


def _all(i):
    return tool_call("background_agents_get_all_tasks", {}, f"l{i}")


def _clear(tid, i):
    return tool_call("background_agents_clear_completed_task", {"task_id": tid}, f"x{i}")


def _start_as(name, i):
    return tool_call("background_agents_start_task",
                     {"agent_name": name, "input": "go", "description": name}, f"s{i}")


def test_declared_subagents_run_on_sheath_provider(tmp_path):
    from agent_framework import SessionContext

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient([text("x")]), bundles=("files",),
                                            subagents={"a": spec([text("r")]),
                                                       "b": spec([text("r")],
                                                                 description="other")})
            providers = list(agent.context_providers)
            ours = [p for p in providers if isinstance(p, DelegationProvider)]
            ctx = SessionContext(input_messages=[])
            await ours[0].before_run(agent=agent, session=agent.create_session(), context=ctx,
                                     state={})
            return providers, ctx

    providers, ctx = asyncio.run(main())
    assert not any(isinstance(p, BackgroundAgentsProvider) for p in providers)
    assert sum(isinstance(p, DelegationProvider) for p in providers) == 1
    assert sorted(t.name for t in ctx.tools) == sorted([
        "background_agents_start_task", "background_agents_wait_for_first_completion",
        "background_agents_get_task_results", "background_agents_get_all_tasks",
        "background_agents_continue_task", "background_agents_clear_completed_task"])
    # Every tool touches the shared task records: none may run in a worker thread.
    assert all(getattr(t, "_invoke_sync_on_event_loop", False) for t in ctx.tools
               if not asyncio.iscoroutinefunction(t.func))
    joined = "\n".join(ctx.instructions)
    assert "background_agents_start_task" in joined
    assert "Available background agents:\n- a: does things\n- b: other" in joined


def test_return_strings_match_maf(tmp_path):
    got: dict = {}
    gate: dict = {}

    async def block() -> str:
        """Wait until the parent releases it."""
        gate.setdefault("event", asyncio.Event())
        await gate["event"].wait()
        return "unblocked"

    async def release() -> str:
        """Release the sub-agent."""
        gate.setdefault("event", asyncio.Event()).set()
        return "released"

    subs = {"a": spec([text("report-a"), tool_call("block"), text("report-2")], tools=[block])}
    script = [
        start("a", 1),                  # s1
        _start_as("zz", 2),             # s2: unknown agent
        wait([], 1),                    # w1: no ids
        wait([1], 2),                   # w2: completes
        wait([1], 3),                   # w3: no longer running
        wait([99], 4),                  # w4: unknown id
        results(1, 1),                  # r1
        results(99, 2),                 # r2
        _all(1),                        # l1
        cont(99, "more", 1),            # c1
        cont(1, "more", 2),             # c2
        cont(1, "again", 3),            # c3: still running
        _clear(1, 1),                   # x1: still running
        results(1, 3),                  # r3: still running
        tool_call("release", call_id="g1"),
        wait([1], 5),                   # w5
        results(1, 4),                  # r4
        _clear(1, 2),                   # x2
        _all(2),                        # l2
        _clear(1, 3),                   # x3
        record(got),
    ]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(script), tools=[release],
                                            bundles=("files",), subagents=subs)
            await agent.run("go", session=agent.create_session())

    asyncio.run(main())
    assert got["s1"] == "Background task 1 started on agent 'a'."
    assert got["s2"] == "Error: No background agent found with name 'zz'. Available agents: a"
    assert got["w1"] == "Error: No task IDs provided."
    assert got["w2"] == "Task 1 finished with status: completed."
    assert got["w3"] == "Task 1 is not running; current status: completed."
    assert got["w4"] == "Error: None of the specified task IDs correspond to running tasks."
    assert got["r1"] == "report-a"
    assert got["r2"] == "Error: No task found with ID 99."
    assert got["l1"] == "Tasks:\n- Task 1 [completed] (a): a"
    assert got["c1"] == "Error: No task found with ID 99."
    assert got["c2"] == "Task 1 continued with new input."
    assert got["c3"] == ("Error: Task 1 is still running. "
                         "Wait for it to complete before continuing.")
    assert got["x1"] == "Error: Task 1 is still running. Wait for it to complete before clearing."
    assert got["r3"] == "Task 1 is still running."
    assert got["w5"] == "Task 1 finished with status: completed."
    assert got["r4"] == "report-2"
    assert got["x2"] == "Task 1 cleared."
    assert got["l2"] == "No tasks."
    assert got["x3"] == "Error: No task found with ID 1."


def test_wait_timeout_string_matches_maf(tmp_path):
    got: dict = {}

    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(1.5)
        return "ok"

    subs = {"a": spec([tool_call("slow"), text("done-a")], tools=[slow])}
    script = [start("a", 1), wait([1], 1), wait([1], 2), results(1, 1), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(script), bundles=("files",),
                                            subagents=subs,
                                            background_agents_wait_timeout_seconds=1)
            await agent.run("go", session=agent.create_session())

    asyncio.run(main())
    assert got["w1"] == ("No background task completed within 1 seconds. The tasks are still "
                         "running; call this tool again if you wish to continue waiting.")
    assert got["w2"] == "Task 1 finished with status: completed."
    assert got["r1"] == "done-a"


def test_task_state_survives_new_agent_session(tmp_path):
    got: dict = {}
    subs = {"a": spec([text("first"), text("second")])}
    # Turn 1 reads its result: an unread finished task would keep the parent loop going.
    turn2_first: list = []
    script = [start("a", 1), wait([1], 1), results(1, 0), text("turn-1"),
              _snap(turn2_first, cont(1, "more", 1)), wait([1], 2), results(1, 1), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(script), bundles=("files",),
                                            subagents=subs)
            first = await agent.run("go", session=agent.create_session())
            await agent.run("more", session=agent.create_session())
            return first.text

    assert asyncio.run(main()) == "turn-1"
    assert got["c1"] == "Task 1 continued with new input."
    assert got["r1"] == "second"
    # The listing is context for the run's first model call (as with the framework's clients).
    assert any("### Current background tasks\n- Task 1 [completed] (a): a" in t
               for t in turn2_first[0])


# --- review fixes: concurrent waits, fresh running state ----------------------------------------

def _with_delegation(tmp_path, subs, body, script=None):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            client = StubChatClient(script or [text("x")])
            agent = await sess.create_agent(client, bundles=("files",), subagents=subs)
            top = sess.delegation.top("data-integrator")
            return await body(sess.delegation, top, agent, client)

    return asyncio.run(main())


def test_parallel_waits_on_one_task_both_report_finished(tmp_path):
    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        return await asyncio.gather(d.wait_first(top, [1]), d.wait_first(top, [1]),
                                    return_exceptions=True)

    out = _with_delegation(tmp_path, {"a": spec([text("report-a")])}, body)
    assert out == ["Task 1 finished with status: completed."] * 2


def test_continue_swapping_future_mid_wait(tmp_path):
    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        first = d._records[0].future
        swapped = {}

        def swap(_fut):
            # Runs before the waiter wakes: read the result, then continue the task.
            swapped["results"] = d.results(top, 1)
            swapped["continue"] = d.continue_task(top, 1, "more")

        first.add_done_callback(swap)
        waited = await d.wait_first(top, [1])
        still = [r.id for r in d.running_owned(top)] if d._records[0].future else []
        again = await d.wait_first(top, [1])
        return waited, swapped, still, again, d.results(top, 1)

    waited, swapped, still, again, final = _with_delegation(
        tmp_path, {"a": spec([text("first"), text("second")])}, body)
    assert swapped == {"results": "first", "continue": "Task 1 continued with new input."}
    assert waited == "Task 1 finished with status: completed."
    assert still == [1]                     # the continued run was not finalized by the wait
    assert again == "Task 1 finished with status: completed."
    assert final == "second"


def test_cancelled_tasks_are_not_running_and_next_turn_does_not_rerun(tmp_path):
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "late"

    subs = {"a": spec([tool_call("slow"), text("never")], tools=[slow])}

    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        await asyncio.sleep(0.05)
        await d.cancel_all()
        running = d.running_owned(top)
        response = await agent.run("next", session=agent.create_session())
        return running, d._records[0].status, response.text, client._turn

    running, status, reply, turns = _with_delegation(
        tmp_path, subs, body, script=[text("x"), text("rerun")])
    assert running == []
    assert status == "failed"
    assert reply == "x"
    assert turns == 1


# --- reach and completion notices ---------------------------------------------------------------

def _notice(tid, path, outcome):
    return f"[task {tid} ({path}) {outcome} — read it with background_agents_get_task_results]"


def _texts(messages):
    return [c.text for m in messages for c in m.contents if c.type == "text" and c.text]


def _snap(calls, then):
    """A script step that records the messages of its model call, then answers ``then``."""
    def step(messages):
        calls.append(_texts(messages) + [str(getattr(c, "result", "")) for m in messages
                                         for c in m.contents if c.type == "function_result"])
        return then
    return step


def _kid(d, name, creator):
    node = AgentNode(name=name, path=name if creator.depth == 0 else f"{creator.path}/{name}",
                     depth=creator.depth + 1, creator=creator)
    d.add_node(node)
    return node


def test_tools_see_only_own_tasks(tmp_path):
    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        await d._records[0].future
        kid = _kid(d, "kid", top)
        return {"results": d.results(kid, 1), "clear": d.clear(kid, 1),
                "continue": d.continue_task(kid, 1, "more"), "all": d.all_tasks(kid),
                "wait": await d.wait_first(kid, [1]), "top_all": d.all_tasks(top)}

    out = _with_delegation(tmp_path, {"a": spec([text("report-a")])}, body)
    assert out["results"] == "Error: No task found with ID 1."
    assert out["clear"] == "Error: No task found with ID 1."
    assert out["continue"] == "Error: No task found with ID 1."
    assert out["all"] == "No tasks."
    assert out["wait"] == "Error: None of the specified task IDs correspond to running tasks."
    assert out["top_all"] == "Tasks:\n- Task 1 [completed] (a): a"


def test_cannot_start_unreachable_agent(tmp_path):
    async def body(d, top, agent, client):
        kid = _kid(d, "kid", top)
        _kid(d, "sib", top)
        _kid(d, "grandkid", kid)
        return (d.start_task(kid, "a", "go", "x"), d.start_task(kid, "SIB", "go", "x"),
                [n.name for n in d.reachable(top)], [n.name for n in d.reachable(kid)],
                d._records)

    role, sib, top_reach, kid_reach, records = _with_delegation(
        tmp_path, {"a": spec([text("r")])}, body)
    assert role == ("Error: No background agent found with name 'a'. "
                    "Available agents: grandkid")
    assert sib == ("Error: No background agent found with name 'SIB'. "
                   "Available agents: grandkid")
    assert top_reach == ["a", "kid", "sib"]
    assert kid_reach == ["grandkid"]
    assert records == []


def test_completion_notice_injected_not_result(tmp_path):
    gate: dict = {}
    calls: list = []

    async def block() -> str:
        """Wait until the parent releases it."""
        await gate.setdefault("event", asyncio.Event()).wait()
        return "unblocked"

    async def tick() -> str:
        """The parent's own work."""
        gate.setdefault("event", asyncio.Event()).set()
        await asyncio.sleep(0.05)
        return "ticked"

    subs = {"a": spec([tool_call("block"), text("SECRET-RESULT")], tools=[block])}
    script = [_snap(calls, start("a", 1)),
              _snap(calls, tool_call("tick", call_id="t1")),
              _snap(calls, tool_call("tick", call_id="t2")),
              _snap(calls, tool_call("tick", call_id="t3")),
              _snap(calls, results(1, 1)),
              _snap(calls, text("done"))]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(script), tools=[tick],
                                            bundles=("files",), subagents=subs)
            await agent.run("go", session=agent.create_session())

    asyncio.run(main())
    notice = _notice(1, "a", "completed")
    before_read, after_read = calls[:5], calls[5:]
    # The notice reaches a model call while the parent is still doing its own work...
    assert any(notice in t for c in calls[1:5] for t in c)
    # ...exactly once, and the result itself stays out of every call until it is read.
    assert all(sum(t == notice for t in c) <= 1 for c in calls)
    assert not any("SECRET-RESULT" in t for c in before_read for t in c)
    assert any("SECRET-RESULT" in t for c in after_read for t in c)


def _run_until_done(tmp_path, subs, cfg=None):
    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        try:
            await d._records[0].future
        except BaseException:
            pass
        await asyncio.sleep(0)     # let the done callback run
        return list(top.inbox)

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r", **(cfg or {}))) as sess:
            await sess.create_agent(StubChatClient([text("x")]), bundles=("files",),
                                    subagents=subs)
            top = sess.delegation.top("data-integrator")
            return await body(sess.delegation, top, None, None)

    return asyncio.run(main())


def test_notice_for_failed_and_timed_out_tasks(tmp_path):
    def boom(messages):
        raise RuntimeError("model down")

    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "late"

    failed = _run_until_done(tmp_path / "f", {"a": spec([boom])})
    timed_out = _run_until_done(tmp_path / "t", {"a": spec([tool_call("slow"), text("never")],
                                                           tools=[slow])},
                                cfg={"subagent_task_timeout_s": 0.05})
    assert failed == [_notice(1, "a", "failed")]
    assert timed_out == [_notice(1, "a", "failed")]


def test_inbox_delivered_when_idle_node_starts(tmp_path):
    calls: list = []
    sub_calls: list = []
    subs = {"a": spec([_snap(sub_calls, text("report-a"))])}

    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        await d._records[0].future
        await asyncio.sleep(0)
        queued = list(top.inbox)
        await agent.run("next", session=agent.create_session())
        # A sub-agent's inbox is delivered when its next task starts.
        d.node("a").inbox.append("hello from the host")
        d.continue_task(top, 1, "more")
        await d._records[0].future
        return queued, list(top.inbox)

    queued, left = _with_delegation(tmp_path, subs, body,
                                    script=[_snap(calls, results(1, 1)), text("done")])
    notice = _notice(1, "a", "completed")
    assert queued == [notice]
    assert any(notice in t for t in calls[0])
    assert left == [notice]          # the first was delivered; this is the continued run's
    assert any("hello from the host" in t for t in sub_calls[-1])


# --- unread finished tasks keep the parent loop going -------------------------------------------

def test_task_finishing_during_final_pass_gets_one_more_pass(tmp_path):
    got: dict = {}
    gate: dict = {}
    client_box: dict = {}

    async def block() -> str:
        """Wait until the parent releases it."""
        await gate.setdefault("event", asyncio.Event()).wait()
        return "unblocked"

    async def release() -> str:
        """Release the sub-agent and let it finish."""
        gate.setdefault("event", asyncio.Event()).set()
        await gate["delegation"]._records[0].future
        await asyncio.sleep(0)
        return "released"

    subs = {"a": spec([tool_call("block"), text("report-a")], tools=[block])}
    script = [start("a", 1), tool_call("release", call_id="g1"),
              text("final-without-reading"),          # the last pass ends unread
              results(1, 1), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            client = StubChatClient(script)
            client_box["client"] = client
            agent = await sess.create_agent(client, tools=[release], bundles=("files",),
                                            subagents=subs)
            gate["delegation"] = sess.delegation
            await agent.run("go", session=agent.create_session())
            return sess.delegation._records[0].unread

    unread = asyncio.run(main())
    assert got["r1"] == "report-a"
    assert unread is False
    # One more pass read it, then the loop ended (no repeats of the last step).
    assert client_box["client"]._turn == len(script)
    nudge = [t for t in _texts(got["messages"]) if "not read" in t]
    assert nudge and "#1 (a)" in nudge[0] and "background_agents_get_task_results" in nudge[0]


def test_reading_results_clears_unread(tmp_path):
    async def body(d, top, agent, client):
        d.start_task(top, "a", "go", "a")
        await d._records[0].future
        await asyncio.sleep(0)
        before = (tasks_running(top)(), tasks_running_message(top)())
        d.results(top, 1)
        after = (tasks_running(top)(), tasks_running_message(top)())
        d.continue_task(top, 1, "more")
        await d._records[0].future
        await asyncio.sleep(0)
        unread_again = d._records[0].unread
        d.clear(top, 1)
        return before, after, unread_again, tasks_running(top)()

    before, after, unread_again, after_clear = _with_delegation(
        tmp_path, {"a": spec([text("first"), text("second")])}, body)
    assert before[0] is True and "#1 (a)" in before[1]
    assert after == (False, None)
    assert unread_again is True
    assert after_clear is False
