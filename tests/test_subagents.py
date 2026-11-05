"""Confined sub-agents: sheath builds them over the parent's Session for MAF's provider (issue #9)."""

from __future__ import annotations

import asyncio
import json

import pytest
from agent_framework import AgentLoopMiddleware, AgentMiddleware

from sheath import SheathConfig, Session, tool_factory
from sheath.status import report_progress
from sheath.subagents import ConfinedSubagent, SubagentSpec
from sheath.testing import StubChatClient, text, tool_call


# --- helpers -----------------------------------------------------------------------------------

def parent(script):
    return StubChatClient(script)


def start(name, i):
    return tool_call("background_agents_start_task",
                     {"agent_name": name, "input": "go", "description": name}, f"s{i}")


def wait(ids, i):
    return tool_call("background_agents_wait_for_first_completion", {"task_ids": ids}, f"w{i}")


def results(tid, i):
    return tool_call("background_agents_get_task_results", {"task_id": tid}, f"r{i}")


def cont(tid, value, i):
    return tool_call("background_agents_continue_task", {"task_id": tid, "text": value}, f"c{i}")


def _result_text(result):
    if isinstance(result, list):
        return "".join(getattr(c, "text", None) or str(c) for c in result)
    return str(result)


def _function_results(messages):
    return [c for m in messages for c in m.contents if c.type == "function_result"]


def echo_results(messages):
    return text(" | ".join(_result_text(c.result) for c in _function_results(messages)))


def capture(into):
    def step(messages):
        for c in _function_results(messages):
            into.append((_result_text(c.result), getattr(c, "exception", None)))
        return text("done")
    return step


def record(into, then=None):
    """Store every function result by call id (and the messages), then answer ``then`` or done."""
    def step(messages):
        into["messages"] = list(messages)
        for c in _function_results(messages):
            into[c.call_id] = _result_text(c.result)
        return then(messages) if then is not None else text("done")
    return step


def spec(script, **kw):
    return SubagentSpec(description=kw.pop("description", "does things"),
                        client=StubChatClient(script), **kw)


def run_parent(tmp_path, script, subagents, *, cfg=None, events=None, tools=None,
               after=None, **agent_kw):
    cfg = cfg or {}
    config = SheathConfig(root_dir=tmp_path / "r", **cfg)

    async def main():
        async with Session.create(config) as sess:
            if events is not None:
                sess.subscribe(events.append)
            agent = await sess.create_agent(parent(script), tools=tools, bundles=("files",),
                                            subagents=subagents, **agent_kw)
            response = await agent.run("go", session=agent.create_session())
            extra = after(sess) if after is not None else None
            return sess, response, extra

    return asyncio.run(main())


# --- acceptance --------------------------------------------------------------------------------

def test_two_subagents_write_files_and_report(tmp_path):
    subs = {
        "a": spec([tool_call("write_file", {"path": "a.txt", "content": "A"}), text("report-a")]),
        "b": spec([tool_call("write_file", {"path": "b.txt", "content": "B"}), text("report-b")]),
    }
    script = [start("a", 1), start("b", 2), wait([1, 2], 1), wait([2], 2),
              results(1, 1), results(2, 2), echo_results]
    sess, response, _ = run_parent(tmp_path, script, subs)
    root = tmp_path / "r"
    assert (root / "a.txt").read_text() == "A"
    assert (root / "b.txt").read_text() == "B"
    assert "report-a" in response.text and "report-b" in response.text
    assert sess.subagent_names == ("a", "b")


def test_subagent_names_default_empty(tmp_path):
    sess = Session.create(SheathConfig(root_dir=tmp_path / "r"))
    assert sess.subagent_names == ()
    assert sess.subagents_running == 0


def test_long_result_arrives_as_handle(tmp_path):
    got: dict = {}
    subs = {"a": spec([text("x" * 500)])}
    script = [start("a", 1), wait([1], 1), results(1, 1), record(got)]
    _, _, handles = run_parent(tmp_path, script, subs, cfg={"spill_threshold_bytes": 64},
                               after=lambda s: dict(s.handles))
    summary = json.loads(got["r1"])
    assert {"id", "kind", "path"} <= summary.keys()
    assert handles[summary["id"]]["source"] == "subagent:a"


def test_result_over_spill_cap_fails_task(tmp_path):
    got: dict = {}
    subs = {"a": spec([text("x" * 500)])}
    script = [start("a", 1), wait([1], 1), results(1, 1), record(got)]
    _, response, _ = run_parent(tmp_path, script, subs,
                                cfg={"spill_threshold_bytes": 64, "max_spill_bytes": 100})
    assert "spill cap" in got["r1"]
    assert response.text == "done"


def test_write_outside_root_fails(tmp_path):
    seen: list = []
    subs = {"a": spec([tool_call("write_file", {"path": "../out.txt", "content": "x"}),
                       capture(seen)])}
    script = [start("a", 1), wait([1], 1), results(1, 1), text("ok")]
    run_parent(tmp_path, script, subs)
    assert not (tmp_path / "out.txt").exists()
    assert "path escapes root" in str(seen[0][1])


def test_background_agents_still_refused(tmp_path):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(StubChatClient([text("x")]), background_agents=[object()])

    with pytest.raises(TypeError, match="background_agents"):
        asyncio.run(main())


def test_subagent_tool_status_carries_name(tmp_path):
    events: list = []

    def ping() -> str:
        """Ping."""
        report_progress("p", tool="ping")
        return "pong"

    def pong() -> str:
        """Pong."""
        report_progress("q", tool="pong")
        return "ping"

    subs = {"a": spec([tool_call("ping"), text("report-a")], tools=[ping])}
    script = [tool_call("pong", call_id="p0"), start("a", 1), wait([1], 1), results(1, 1),
              text("ok")]
    run_parent(tmp_path, script, subs, events=events, tools=[pong])
    pings = [e for e in events if e.tool == "ping"]
    assert pings and all(e.agent == "a" for e in pings)
    assert any(e.message == "p" for e in pings)
    sheath_events = [(e.message, e.agent) for e in events if e.tool == "sheath"]
    assert ("started", "a") in sheath_events
    assert ("finished", "a") in sheath_events
    pongs = [e for e in events if e.tool == "pong"]
    assert pongs and all(e.agent is None for e in pongs)


def test_every_subagent_tool_call_reports_activity(tmp_path):
    events: list = []
    subs = {"a": spec([tool_call("write_file", {"path": "ok.txt", "content": "1"}, "1"),
                       tool_call("write_file", {"path": "../x", "content": "2"}, "2"),
                       text("report-a")])}
    script = [start("a", 1), wait([1], 1), results(1, 1), text("ok")]
    run_parent(tmp_path, script, subs, events=events)
    seq = [(e.tool, e.message, e.agent) for e in events]
    expected = [("write_file", "running", "a"), ("write_file", "done", "a"),
                ("write_file", "running", "a"),
                ("write_file", "failed: PathEscapesRootError", "a")]
    activity = [s for s in seq if s[0] == "write_file"]
    assert activity == expected
    assert not any("../x" in e.message for e in events if e.message.startswith("failed"))
    assert not any(e.tool.startswith("background_agents") for e in events)


def test_cap_one_serializes(tmp_path):
    log: list = []

    async def step(tag: str) -> str:
        """Do a step."""
        log.append(("start", tag))
        await asyncio.sleep(0.05)
        log.append(("end", tag))
        return "ok"

    subs = {"a": spec([tool_call("step", {"tag": "a"}), text("ra")], tools=[step]),
            "b": spec([tool_call("step", {"tag": "b"}), text("rb")], tools=[step])}
    script = [start("a", 1), start("b", 2), wait([1, 2], 1), wait([1, 2], 2),
              results(1, 1), results(2, 2), text("ok")]
    run_parent(tmp_path, script, subs, cfg={"subagent_max_concurrent": 1})
    assert len(log) == 4
    x, y = log[0][1], log[2][1]
    assert log == [("start", x), ("end", x), ("start", y), ("end", y)]
    assert {x, y} == {"a", "b"}


def test_task_timeout_fails_with_message(tmp_path):
    got: dict = {}

    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(2)
        return "late"

    subs = {"a": spec([tool_call("slow"), text("never")], tools=[slow])}
    script = [start("a", 1), wait([1], 1), results(1, 1), record(got)]
    run_parent(tmp_path, script, subs, cfg={"subagent_task_timeout_s": 0.05})
    assert "ran past subagent_task_timeout_s" in got["r1"]


def test_continue_task_goes_through_wrapper(tmp_path):
    got: dict = {}
    events: list = []
    subs = {"a": spec([text("short"), text("y" * 500)])}
    script = [start("a", 1), wait([1], 1), cont(1, "more", 1), wait([1], 2), results(1, 1),
              record(got)]
    run_parent(tmp_path, script, subs, cfg={"spill_threshold_bytes": 64}, events=events)
    summary = json.loads(got["r1"])
    assert {"id", "kind", "path"} <= summary.keys()
    started = [e for e in events if e.tool == "sheath" and e.message == "started"]
    assert [e.agent for e in started] == ["a", "a"]


def test_tool_factory_built_once_per_subagent(tmp_path):
    calls: list = []

    def build():
        calls.append(1)

        def helper() -> str:
            """Help."""
            return "h"
        return helper

    factory = tool_factory(build)

    async def main():
        for i in range(2):
            cfg = SheathConfig(root_dir=tmp_path / f"r{i}")
            async with Session.create(cfg) as sess:
                await sess.create_agent(
                    StubChatClient([text("x")]), bundles=("files",),
                    subagents={"a": spec([text("r")], tools=[factory])})

    asyncio.run(main())
    assert len(calls) == 2


def test_subagent_mcp_connected_tagged_and_closed(tmp_path, monkeypatch):
    from tests.test_mcp_wiring import FakeMCPTool

    import sheath.mcp_status as mcp_status

    agents_seen: list = []
    real = mcp_status.install_status_wrappers

    def spy(bus, tool, server, *, agent=None):
        agents_seen.append(agent)
        return real(bus, tool, server, agent=agent)

    monkeypatch.setattr(mcp_status, "install_status_wrappers", spy)
    mcp = FakeMCPTool()

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(StubChatClient([text("x")]), bundles=("files",),
                                    subagents={"a": spec([text("r")], tools=[mcp])})
            assert mcp.connected
            assert not mcp.closed

    asyncio.run(main())
    assert mcp.closed
    assert agents_seen == ["a"]


def test_budget_applied_to_own_client_only(tmp_path):
    sub_client = StubChatClient([text("r")])
    parent_client = StubChatClient([text("x")])
    before = parent_client.function_invocation_configuration["max_iterations"]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(
                parent_client, bundles=("files",),
                subagents={"a": SubagentSpec(description="d", client=sub_client,
                                             max_round_trips=2)})

    asyncio.run(main())
    assert sub_client.function_invocation_configuration["max_iterations"] == 2
    assert parent_client.function_invocation_configuration["max_iterations"] == before


def test_parent_rerun_until_tasks_finish(tmp_path):
    got: dict = {}

    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(0.2)
        return "ok"

    subs = {"a": spec([tool_call("slow"), text("report-a")], tools=[slow])}
    script = [start("a", 1), text("finished early"), wait([1], 1), results(1, 1),
              record(got, echo_results)]
    _, response, _ = run_parent(tmp_path, script, subs)
    assert "report-a" in response.text
    assert "finished early" not in response.text
    texts = [c.text for m in got["messages"] for c in m.contents if c.type == "text" and c.text]
    assert any("still have 1 background task" in t for t in texts)


class _Mine(AgentMiddleware):
    async def process(self, context, call_next):
        await call_next()


def _loops(agent):
    return [m for m in agent.middleware if isinstance(m, AgentLoopMiddleware)]


def _build(tmp_path, **agent_kw):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            return await sess.create_agent(StubChatClient([text("x")]), bundles=("files",),
                                           subagents={"a": spec([text("r")])}, **agent_kw)

    return asyncio.run(main())


def test_caller_loop_replaces_sheaths(tmp_path):
    mine = _Mine()

    def stop(**kw):
        return False

    agent = _build(tmp_path, loop_should_continue=stop)
    loops = _loops(agent)
    assert len(loops) == 1 and loops[0].should_continue is stop   # only the caller's (MAF-built)

    agent = _build(tmp_path, middleware=[mine])
    loops = _loops(agent)
    assert len(loops) == 1 and loops[0].return_final_only is True
    order = list(agent.middleware)
    mine_idx = next(i for i, m in enumerate(order) if isinstance(m, _Mine))
    assert order.index(loops[0]) < mine_idx


def test_caller_loop_max_iterations_used(tmp_path):
    agent = _build(tmp_path, loop_max_iterations=3)
    loops = _loops(agent)
    assert len(loops) == 1 and loops[0].max_iterations == 3


def test_no_loop_without_subagents(tmp_path):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            return await sess.create_agent(StubChatClient([text("x")]), bundles=("files",))

    assert _loops(asyncio.run(main())) == []


class _ToolRecordingStub(StubChatClient):
    def __init__(self, script):
        super().__init__(script)
        self.tool_names: list[set] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.tool_names.append({getattr(t, "name", None) for t in (options.get("tools") or [])})
        return super()._inner_get_response(messages=messages, stream=stream, options=options,
                                           **kwargs)


def test_subagent_cannot_delegate(tmp_path):
    seen: list = []
    events: list = []
    sub_client = _ToolRecordingStub([start("a", 9), capture(seen)])
    subs = {"a": SubagentSpec(description="d", client=sub_client)}
    script = [start("a", 1), wait([1], 1), results(1, 1), text("ok")]
    run_parent(tmp_path, script, subs, events=events)
    # The sub-agent was never offered any delegation tool ...
    assert sub_client.tool_names and all(
        not any((n or "").startswith("background_agents") for n in names)
        for names in sub_client.tool_names)
    assert "write_file" in sub_client.tool_names[0]
    # ... its attempt came back as an error, and no second task ever started.
    assert seen and (seen[0][1] is not None or "rror" in seen[0][0])
    started = [e for e in events if e.tool == "sheath" and e.message == "started"]
    assert len(started) == 1


def test_stream_true_refused():
    sub = ConfinedSubagent(object(), name="a", description="d", session=None)
    with pytest.raises(ValueError, match="stream"):
        asyncio.run(sub.run("go", stream=True))
