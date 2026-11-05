"""Sub-agents over AG-UI: live sub-agent status, no loop nudges in the chat, cross-request tasks."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest
from agent_framework import AgentLoopMiddleware

pytest.importorskip("agent_framework_ag_ui")

from sheath import Session, Sheath, SheathConfig, SubagentSpec  # noqa: E402
from sheath.status import report_progress  # noqa: E402
from sheath.subagents import _DropLoopNudges  # noqa: E402
from sheath.testing import StubChatClient, text, tool_call  # noqa: E402

_NUDGE = "background task(s) running"


def start(name, i):
    return tool_call("background_agents_start_task",
                     {"agent_name": name, "input": "go", "description": name}, f"s{i}")


def wait(ids, i):
    return tool_call("background_agents_wait_for_first_completion", {"task_ids": ids}, f"w{i}")


def results(tid, i):
    return tool_call("background_agents_get_task_results", {"task_id": tid}, f"r{i}")


def cont(tid, value, i):
    return tool_call("background_agents_continue_task", {"task_id": tid, "text": value}, f"c{i}")


def _cfg(tmp_path) -> SheathConfig:
    cfg = SheathConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"   # skip the dotenv lookup
    return cfg


def _dump(ev) -> dict:
    return ev.model_dump(exclude_none=True) if hasattr(ev, "model_dump") else {}


def _type(d) -> str:
    t = d.get("type")
    return getattr(t, "value", t)


def _function_results(messages):
    return {c.call_id: c for m in messages for c in m.contents if c.type == "function_result"}


def _result_text(result):
    if isinstance(result, list):
        return "".join(getattr(c, "text", None) or str(c) for c in result)
    return str(result)


async def _collect(sheath, input_data, on_event=None):
    out = []
    async for ev in sheath.agui_stream(input_data):
        d = _dump(ev)
        out.append(d)
        if on_event is not None:
            on_event(d)
    return out


def test_subagent_status_streams_before_wait_result(tmp_path):
    # The sub's tool reports progress and then blocks until the consumer has seen that status.
    # Before the fix, status was only flushed when the parent's stream produced an event, which
    # it cannot do while it sits in wait_for_first_completion: the gate would time out.
    async def main():
        gate = asyncio.Event()
        released_by_consumer = []

        async def slow() -> str:
            """Do slow work."""
            report_progress("halfway", tool="slow")
            try:
                await asyncio.wait_for(gate.wait(), timeout=3)
                released_by_consumer.append(True)
            except TimeoutError:
                released_by_consumer.append(False)
            return "ok"

        subs = {"a": SubagentSpec(description="slow agent", tools=(slow,),
                                  client=StubChatClient([tool_call("slow"), text("sub-result")]))}
        parent = StubChatClient([start("a", 1), wait([1], 1), results(1, 1), text("final")])
        sheath = Sheath(_cfg(tmp_path), client=parent, bundles=("files",), subagents=subs)

        def on_event(d):
            if (_type(d) == "CUSTOM" and d.get("name") == "sheath.status"
                    and d["value"].get("message") == "halfway"):
                gate.set()

        try:
            events = await _collect(sheath, {"threadId": "t1", "runId": "r1",
                                             "messages": [{"id": "u1", "role": "user",
                                                           "content": "go"}]}, on_event)
        finally:
            await sheath.aclose_sessions()
        return events, released_by_consumer

    events, released = asyncio.run(main())
    assert released == [True]
    status_i = next(i for i, d in enumerate(events)
                    if _type(d) == "CUSTOM" and d.get("name") == "sheath.status"
                    and d["value"].get("message") == "halfway")
    assert events[status_i]["value"]["agent"] == "a"
    wait_result_i = next(i for i, d in enumerate(events)
                         if _type(d) == "TOOL_CALL_RESULT" and d.get("tool_call_id") == "w1")
    assert status_i < wait_result_i


def test_nudge_absent_from_text_and_snapshot(tmp_path):
    async def main():
        gate = asyncio.Event()

        async def slow() -> str:
            """Do slow work."""
            await asyncio.wait_for(gate.wait(), timeout=3)
            return "ok"

        subs = {"a": SubagentSpec(description="slow agent", tools=(slow,),
                                  client=StubChatClient([tool_call("slow"), text("sub-result")]))}
        parent = StubChatClient([start("a", 1), text("finished early"), wait([1], 1),
                                 results(1, 1), text("final")])
        sheath = Sheath(_cfg(tmp_path), client=parent, bundles=("files",), subagents=subs)

        def on_event(d):
            # Release the sub only once the parent is waiting, so the nudge pass happens.
            if _type(d) == "TOOL_CALL_START" and d.get("tool_call_id") == "w1":
                gate.set()

        try:
            return await _collect(sheath, {"threadId": "t1", "runId": "r1",
                                           "messages": [{"id": "u1", "role": "user",
                                                         "content": "go"}]}, on_event)
        finally:
            await sheath.aclose_sessions()

    events = asyncio.run(main())
    deltas = [d.get("delta", "") for d in events if _type(d) == "TEXT_MESSAGE_CONTENT"]
    snapshot_texts = [str(m.get("content", "")) for d in events
                      if _type(d) == "MESSAGES_SNAPSHOT" for m in d.get("messages", [])]
    assert not any(_NUDGE in t for t in deltas)
    assert not any(_NUDGE in t for t in snapshot_texts)
    assert any("finished early" in t for t in deltas)   # real model output stays
    assert any("final" in t for t in deltas)


def test_nonstreaming_run_unaffected(tmp_path):
    async def main():
        gate = asyncio.Event()

        async def slow() -> str:
            """Do slow work."""
            await asyncio.wait_for(gate.wait(), timeout=3)
            return "ok"

        def release_then_wait(messages):
            gate.set()
            return wait([1], 1)

        subs = {"a": SubagentSpec(description="slow agent", tools=(slow,),
                                  client=StubChatClient([tool_call("slow"), text("sub-result")]))}
        parent = StubChatClient([start("a", 1), text("finished early"), release_then_wait,
                                 results(1, 1), text("final")])
        async with Session.create(_cfg(tmp_path)) as sess:
            agent = await sess.create_agent(parent, bundles=("files",), subagents=subs)
            order = list(agent.middleware)
            loop_i = next(i for i, m in enumerate(order) if isinstance(m, AgentLoopMiddleware))
            assert loop_i > 0 and isinstance(order[loop_i - 1], _DropLoopNudges)
            response = await agent.run("go", session=agent.create_session())
            return response.text

    assert asyncio.run(main()) == "final"


def test_continue_task_across_agui_requests(tmp_path):
    """OBSERVED: a task from an earlier AG-UI request can NOT be continued: "No task found".

    Request 1 starts task 1 and waits for it; request 2 on the same ``threadId`` calls
    ``continue_task(1, ...)`` and gets "Error: No task found with ID 1.". The AG-UI wrapper
    builds a fresh ``AgentSession`` (same ``session_id``, the ``threadId``) with empty state for
    every request, and MAF's provider keeps its task list in that session state, so each
    request starts with no tasks (and task ids start again at 1).
    """
    async def main():
        seen = {}

        def record(messages):
            for call_id, c in _function_results(messages).items():
                seen[call_id] = _result_text(c.result)
            return text("done")

        subs = {"a": SubagentSpec(description="agent",
                                  client=StubChatClient([text("first"), text("second")]))}
        parent = StubChatClient([start("a", 1), wait([1], 1), results(1, 1), text("r1-done"),
                                 cont(1, "more", 1), record])
        sheath = Sheath(_cfg(tmp_path), client=parent, bundles=("files",), subagents=subs)
        try:
            await _collect(sheath, {"threadId": "t1", "runId": "r1",
                                    "messages": [{"id": "u1", "role": "user", "content": "go"}]})
            await _collect(sheath, {"threadId": "t1", "runId": "r2",
                                    "messages": [{"id": "u1", "role": "user", "content": "go"},
                                                 {"id": "a1", "role": "assistant",
                                                  "content": "r1-done"},
                                                 {"id": "u2", "role": "user", "content": "more"}]})
        finally:
            await sheath.aclose_sessions()
        return seen

    seen = asyncio.run(main())
    assert seen["c1"] == "Error: No task found with ID 1."


def test_client_disconnect_stops_the_run_and_its_subagents(tmp_path):
    # Closing the stream while the parent sits in wait_for_first_completion must stop the
    # parent's run (it makes no further model call, so it cannot start another task after the
    # turn-end cancel has taken its snapshot) and cancel the running sub-agent.
    async def main():
        never = asyncio.Event()
        parent_waiting = asyncio.Event()
        sub_outcome = []

        async def slow() -> str:
            """Do slow work."""
            await asyncio.wait_for(parent_waiting.wait(), timeout=3)
            await asyncio.sleep(0.05)              # the parent enters its wait tool meanwhile
            report_progress("halfway", tool="slow")
            try:
                await asyncio.wait_for(never.wait(), timeout=3)
            except asyncio.CancelledError:
                sub_outcome.append("cancelled")
                raise
            except TimeoutError:
                sub_outcome.append("timed out")
            return "ok"

        subs = {"a": SubagentSpec(description="slow agent", tools=(slow,),
                                  client=StubChatClient([tool_call("slow"), text("sub-result")]))}
        parent = StubChatClient([start("a", 1), wait([1], 1), start("a", 2), wait([2], 2),
                                 text("final")])
        sheath = Sheath(_cfg(tmp_path), client=parent, bundles=("files",), subagents=subs)
        try:
            async with aclosing(sheath.agui_stream(
                    {"threadId": "t1", "runId": "r1",
                     "messages": [{"id": "u1", "role": "user", "content": "go"}]})) as stream:
                async for ev in stream:
                    d = _dump(ev)
                    if _type(d) == "TOOL_CALL_START" and d.get("tool_call_id") == "w1":
                        parent_waiting.set()
                    if (_type(d) == "CUSTOM" and d.get("name") == "sheath.status"
                            and d["value"].get("message") == "halfway"):
                        break                      # the client disconnects mid-wait
            turns_at_close = parent._turn
            await asyncio.sleep(0.2)               # give a still-running parent time to go on
            conv = sheath._sessions().get("t1")
            return turns_at_close, parent._turn, conv.session.subagents_running, sub_outcome
        finally:
            await sheath.aclose_sessions()

    turns_at_close, turns_after, running, sub_outcome = asyncio.run(main())
    assert turns_at_close == 2                     # start, wait: then the client left
    assert turns_after == 2                        # the parent made no further model call
    assert running == 0
    assert sub_outcome == ["cancelled"]
