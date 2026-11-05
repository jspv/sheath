"""``subagent_middleware``: host middleware for every sub-agent, declared and spawned."""

from __future__ import annotations

import asyncio
import warnings

import pytest
from agent_framework import FunctionMiddleware

from sheath import Sheath, SheathConfig, Session
from sheath.subagents import SpawnPolicy, SubagentSpec
from sheath.testing import text, tool_call

from tests.test_spawn import Rec, spawn
from tests.test_spawn_nesting import Queue
from tests.test_subagents import record, results, start, wait


class Guard(FunctionMiddleware):
    """Blocks write_file; records every tool call it sees, tagged with the sub-agent path."""

    def __init__(self, path: str, seen: list):
        self.path = path
        self.seen = seen

    async def process(self, context, call_next):
        self.seen.append((self.path, context.function.name))
        if context.function.name == "write_file":
            context.result = "Error: blocked by host policy"
            return
        await call_next()


def factory_for(seen: list, made: list):
    def factory(path: str):
        g = Guard(path, seen)
        made.append(g)
        return [g]
    return factory


def _write(path, i):
    return tool_call("write_file", {"path": path, "content": "c"}, f"wf{i}")


async def _run(tmp_path, script, *, subagents=None, spawn_policy=None, factory=None,
               sub_mw=None, middleware=None):
    async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
        kw = {} if middleware is None else {"middleware": middleware}
        agent = await s.create_agent(Rec(script), bundles=("files",), subagents=subagents,
                                     spawn=spawn_policy, client_factory=factory,
                                     subagent_middleware=sub_mw, **kw)
        await agent.run("go", session=agent.create_session())
        return s


def test_spawned_subagent_cannot_escape_host_guard(tmp_path):
    seen: list = []
    made: list = []
    a_got: dict = {}
    a = [_write("agents/a/x.md", 1), record(a_got, lambda m: text("a-done"))]
    script = [spawn("a", 1), wait([1], 1), results(1, 1), text("done")]
    s = asyncio.run(asyncio.wait_for(_run(
        tmp_path, script, spawn_policy=SpawnPolicy(models=("m",)), factory=Queue(a),
        sub_mw=factory_for(seen, made)), 5))
    assert a_got["wf1"] == "Error: blocked by host policy"
    assert not (s.root / "agents/a/x.md").exists()
    assert ("a", "write_file") in seen


def test_declared_role_gets_it(tmp_path):
    seen: list = []
    made: list = []
    role = Rec([_write("agents/role/x.md", 1), text("role-done")])
    subs = {"role": SubagentSpec(description="d", client=role)}
    script = [start("role", 1), wait([1], 1), results(1, 1), text("done")]
    s = asyncio.run(asyncio.wait_for(_run(tmp_path, script, subagents=subs,
                                          sub_mw=factory_for(seen, made)), 5))
    assert ("role", "write_file") in seen
    assert not (s.root / "agents/role/x.md").exists()


def test_fresh_instances_per_subagent_nested_included(tmp_path):
    seen: list = []
    made: list = []
    lead = [spawn("child", 1), wait([2], 1), results(2, 1), text("lead-done")]
    child = [text("child-done")]
    script = [spawn("lead", 1, tools=["spawn_subagent"]), spawn("solo", 2),
              wait([1], 1), wait([3], 2), results(1, 1), results(3, 2), text("done")]
    asyncio.run(asyncio.wait_for(_run(
        tmp_path, script, spawn_policy=SpawnPolicy(models=("m",)),
        factory=Queue(lead, child, [text("solo-done")]), sub_mw=factory_for(seen, made)), 5))
    assert sorted(g.path for g in made) == ["lead", "lead/child", "solo"]
    assert len({id(g) for g in made}) == 3
    # The parent's own calls are not seen by sub-agent middleware.
    assert all(p in {"lead", "lead/child", "solo"} for p, _ in seen)


def test_factory_failure_refuses_spawn(tmp_path):
    got: dict = {}

    def broken(path):
        raise RuntimeError("no")

    script = [spawn("a", 1), record(got)]
    s = asyncio.run(asyncio.wait_for(_run(
        tmp_path, script, spawn_policy=SpawnPolicy(models=("m",)), factory=Queue([text("a")]),
        sub_mw=broken), 5))
    assert got["sp1"] == "Error: could not build sub-agent 'a' (RuntimeError)."
    assert s.delegation.node("a") is None


def test_factory_must_return_a_sequence(tmp_path):
    got: dict = {}
    script = [spawn("a", 1), record(got)]
    asyncio.run(asyncio.wait_for(_run(
        tmp_path, script, spawn_policy=SpawnPolicy(models=("m",)), factory=Queue([text("a")]),
        sub_mw=lambda path: Guard(path, [])), 5))
    assert got["sp1"].startswith("Error: could not build sub-agent 'a'")

    subs = {"role": SubagentSpec(description="d", client=Rec([text("x")]))}
    with pytest.raises(TypeError, match="subagent_middleware"):
        asyncio.run(_run(tmp_path / "2", [text("x")], subagents=subs,
                         sub_mw=lambda path: Guard(path, [])))


def test_must_be_callable(tmp_path):
    subs = {"role": SubagentSpec(description="d", client=Rec([text("x")]))}
    with pytest.raises(TypeError, match="subagent_middleware"):
        asyncio.run(_run(tmp_path, [text("x")], subagents=subs, sub_mw=[Guard("x", [])]))


def test_warns_when_parent_middleware_would_not_reach_subagents(tmp_path):
    subs = {"role": SubagentSpec(description="d", client=Rec([text("x")]))}
    with pytest.warns(UserWarning, match="subagent_middleware"):
        asyncio.run(_run(tmp_path / "1", [text("x")], subagents=subs,
                         middleware=[Guard("top", [])]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # given a factory: no warning
        asyncio.run(_run(tmp_path / "2", [text("x")], subagents=subs,
                         middleware=[Guard("top", [])], sub_mw=lambda p: []))
        # no sub-agents: nothing to warn about
        asyncio.run(_run(tmp_path / "3", [text("x")], middleware=[Guard("top", [])]))
    assert not any("subagent_middleware" in str(w.message) for w in caught)


def test_one_factory_per_session(tmp_path):
    q = Queue()

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
            await s.create_agent(Rec([text("x")]), bundles=("files",),
                                 spawn=SpawnPolicy(models=("m",)), client_factory=q,
                                 subagent_middleware=lambda p: [])
            await s.create_agent(Rec([text("x")]), bundles=("files",),
                                 spawn=SpawnPolicy(models=("m",)), client_factory=q,
                                 subagent_middleware=lambda p: [], name="other")

    with pytest.raises(ValueError, match="subagent_middleware"):
        asyncio.run(main())


def test_sheath_passes_it_to_one_shots_and_conversations(tmp_path):
    made: list = []
    seen: list = []
    factory = factory_for(seen, made)

    def make_sheath(script, sub_script):
        return Sheath(SheathConfig(root_dir=tmp_path / "r"), client=Rec(script),
                      bundles=("files",), spawn=SpawnPolicy(models=("m",)),
                      client_factory=Queue(sub_script), subagent_middleware=factory)

    script = [spawn("a", 1), wait([1], 1), results(1, 1), text("done")]
    make_sheath(script, [_write("agents/a/x.md", 1), text("a")]).solve("go")
    assert [g.path for g in made] == ["a"]

    async def conv():
        h = make_sheath(script, [_write("agents/a/x.md", 1), text("a")])
        c = await h.aopen("c1")
        try:
            await c.aask("go")
        finally:
            await c.aclose()

    asyncio.run(conv())
    assert [g.path for g in made] == ["a", "a"]
