"""sheath's own delegation provider: parity with MAF's background-agents tools."""

from __future__ import annotations

import asyncio

from agent_framework import BackgroundAgentsProvider

from sheath import SheathConfig, Session
from sheath.delegation import DelegationProvider
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
    script = [start("a", 1), wait([1], 1), text("turn-1"),
              cont(1, "more", 1), wait([1], 2), results(1, 1), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(StubChatClient(script), bundles=("files",),
                                            subagents=subs)
            await agent.run("go", session=agent.create_session())
            await agent.run("more", session=agent.create_session())

    asyncio.run(main())
    assert got["c1"] == "Task 1 continued with new input."
    assert got["r1"] == "second"
    texts = [c.text for m in got["messages"] for c in m.contents if c.type == "text" and c.text]
    assert any("### Current background tasks\n- Task 1 [completed] (a): a" in t for t in texts)


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
