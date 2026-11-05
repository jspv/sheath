"""Runtime sub-agents: ``SpawnPolicy`` and the top agent's ``spawn_subagent`` tool."""

from __future__ import annotations

import asyncio

import pytest

from agent_framework import ChatResponse, Message

from sheath import SheathConfig, Session
from sheath.subagents import SpawnPolicy, SubagentSpec, validate_spawn_policy
from sheath.testing import StubChatClient, text, tool_call

from tests.test_subagents import record, results, start, wait

# --- helpers -----------------------------------------------------------------------------------


class Rec(StubChatClient):
    """A stub client that records, per model call, the tools it was offered (name -> description)."""

    def __init__(self, script, model=None):
        super().__init__(script)
        self.model = model
        self.calls: list[dict[str, str]] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls.append({t.name: (t.description or "") for t in options.get("tools") or []})
        step = self._script[min(self._turn, len(self._script) - 1)] if self._script else None
        if isinstance(step, list):   # several tool calls in one model turn
            self._turn += 1

            async def _response():
                return ChatResponse(messages=Message(role="assistant", contents=list(step)))
            return _response()
        return super()._inner_get_response(messages=messages, stream=stream, options=options,
                                           **kwargs)


class Factory:
    """A client factory recording each model it is asked for; scripts are chosen by model."""

    def __init__(self, scripts=None):
        self.scripts = scripts or {}
        self.made: list[tuple[str, Rec]] = []

    def __call__(self, model):
        client = Rec(self.scripts.get(model, [text(f"report from {model}")]), model=model)
        self.made.append((model, client))
        return client


def spawn(name, i, **kw):
    args = {"name": name, "instructions": "You chase one lead.", "task": "go", **kw}
    return tool_call("spawn_subagent", args, f"sp{i}")


def tool_a(q: str) -> str:
    """Host tool A."""
    return f"a:{q}"


def tool_b(q: str) -> str:
    """Host tool B."""
    return f"b:{q}"


POLICY = SpawnPolicy(models=("m-small", "m-big"))


def snapshot(sess):
    d = sess.delegation
    agents = sess.root / "agents"
    return {
        "nodes": sorted(n.name for n in d._nodes.values()),
        "records": len(d._records),
        "total": d.spawned_total,
        "folders": sorted(p.name for p in agents.iterdir()) if agents.exists() else [],
    }


def run_spawn(tmp_path, script, *, spawn_policy=POLICY, factory=None, subagents=None,
              tools=None, bundles=("files",), cfg=None, parent_client=None, runs=1):
    config = SheathConfig(root_dir=tmp_path / "r", **(cfg or {}))
    factory = factory if factory is not None else Factory()
    parent = parent_client or Rec(script)

    async def main():
        async with Session.create(config) as sess:
            agent = await sess.create_agent(parent, tools=tools, bundles=bundles,
                                            subagents=subagents, spawn=spawn_policy,
                                            client_factory=factory)
            agent_session = agent.create_session()
            for _ in range(runs):
                await agent.run("go", session=agent_session)
            return sess, snapshot(sess)

    sess, snap = asyncio.run(main())
    return sess, snap, factory, parent


NOTHING = {"nodes": [], "records": 0, "total": 0, "folders": []}


# --- acceptance --------------------------------------------------------------------------------

def test_spawn_two_with_different_models_and_tools(tmp_path):
    factory = Factory({
        "m-small": [tool_call("write_file", {"path": "a.txt", "content": "A"}), text("report-a")],
        "m-big": [tool_call("write_file", {"path": "b.txt", "content": "B"}), text("report-b")],
    })
    got: dict = {}
    script = [
        spawn("a", 1, model="m-small", tools=["tool_a"], bundles=["files"]),
        spawn("b", 2, model="m-big", tools=["tool_b"]),
        wait([1, 2], 1), wait([2], 2), results(1, 1), results(2, 2),
        tool_call("read_file", {"path": "a.txt"}, "f1"),
        tool_call("read_file", {"path": "b.txt"}, "f2"),
        record(got),
    ]
    sess, snap, factory, _ = run_spawn(tmp_path, script, factory=factory,
                                       tools=[tool_a, tool_b], bundles=("files", "code"))
    assert got["sp1"] == "Background task 1 started on agent 'a'."
    assert got["sp2"] == "Background task 2 started on agent 'b'."
    assert [m for m, _ in factory.made] == ["m-small", "m-big"]
    assert "report-a" in got["r1"] and "report-b" in got["r2"]
    assert "A" in got["f1"] and "B" in got["f2"]
    files = {"read_file", "write_file", "edit_file", "list_files", "search", "inspect_handle"}
    small, big = factory.made[0][1], factory.made[1][1]
    # send_message: every agent has it when the host's policy is on (messaging).
    assert small.calls and all(set(c) == files | {"tool_a", "send_message"} for c in small.calls)
    assert big.calls and all(set(c) == files | {"run_python", "tool_b", "send_message"}
                             for c in big.calls)
    assert snap == {"nodes": ["a", "b"], "records": 2, "total": 2, "folders": ["a", "b"]}
    d = sess.delegation
    top = d.top("data-integrator")
    # Handed down as the creator's own built objects, never re-wrapped.
    assert d.node("a").tool_objects["tool_a"] is top.tool_objects["tool_a"]
    assert d.node("b").tool_objects["tool_b"] is top.tool_objects["tool_b"]
    assert d.node("a").creator is top and d.node("a").depth == 1
    assert d.node("a").bundles == ("files",)
    assert not d.node("a").can_spawn


def test_spawn_hands_down_connected_mcp_function(tmp_path):
    from tests.test_mcp_wiring import FakeMCPTool

    connects = []

    class CountingMCP(FakeMCPTool):
        async def connect(self):
            connects.append(1)
            await super().connect()

    mcp = CountingMCP()
    factory = Factory({"m-small": [tool_call("mcp_query", {"q": "x"}), text("ok")]})
    got: dict = {}
    script = [spawn("a", 1, tools=["mcp_query"]), wait([1], 1), results(1, 1), record(got)]
    sess, _, factory, _ = run_spawn(tmp_path, script, factory=factory, tools=[mcp])
    assert connects == [1]
    top = sess.delegation.top("data-integrator")
    assert sess.delegation.node("a").tool_objects["mcp_query"] is mcp.functions[0]
    assert top.tool_objects["mcp_query"] is mcp.functions[0]
    assert "mcp_query" in factory.made[0][1].calls[0]


def test_spawned_agent_startable_by_name(tmp_path):
    got: dict = {}
    factory = Factory({"m-small": [text("first"), text("second")]})
    script = [spawn("a", 1), wait([1], 1), results(1, 1),
              start("a", 2), wait([2], 2), results(2, 2), record(got)]
    _, snap, _, _ = run_spawn(tmp_path, script, factory=factory)
    assert got["s2"] == "Background task 2 started on agent 'a'."
    assert got["r1"] == "first" and got["r2"] == "second"
    assert snap["total"] == 1 and snap["records"] == 2


def test_spawned_folder_created(tmp_path):
    script = [spawn("lead-2", 1), wait([1], 1), results(1, 1), text("done")]
    sess, snap, _, _ = run_spawn(tmp_path, script)
    assert snap["folders"] == ["lead-2"]
    assert sess.delegation.node("lead-2").folder == "agents/lead-2"


def test_no_policy_no_tool(tmp_path):
    for subagents in (None, {"x": SubagentSpec(description="d",
                                               client=StubChatClient([text("x")]))}):
        parent = Rec([text("done")])
        run_spawn(tmp_path / str(bool(subagents)), [], spawn_policy=None, factory=None,
                  subagents=subagents, parent_client=parent)
        assert parent.calls and all("spawn_subagent" not in c for c in parent.calls)


def test_tool_description_shows_models_and_budget(tmp_path):
    policy = SpawnPolicy(models=("m-small", "m-big"), default_model="m-big", max_subagents=3)
    script = [spawn("a", 1), wait([1], 1), results(1, 1), text("first run"), text("second run")]
    _, _, _, parent = run_spawn(tmp_path, script, spawn_policy=policy, runs=2)
    first = parent.calls[0]["spawn_subagent"]
    last = parent.calls[-1]["spawn_subagent"]
    for desc in (first, last):
        assert "m-small" in desc and "m-big" in desc
        assert "default model: m-big" in desc.lower()
    assert "3 of 3" in first
    assert "2 of 3" in last


# --- policy validation -------------------------------------------------------------------------

def _factory(model):
    return StubChatClient([text("x")])


@pytest.mark.parametrize("policy, factory, match", [
    (SpawnPolicy(models=()), _factory, "models"),
    (SpawnPolicy(models=("m", "")), _factory, "model"),
    (SpawnPolicy(models=("m", 3)), _factory, "model"),
    (SpawnPolicy(models=("m",), default_model="other"), _factory, "default_model"),
    (SpawnPolicy(models=("m",), max_subagents=0), _factory, "max_subagents"),
    (SpawnPolicy(models=("m",), max_depth=0), _factory, "max_depth"),
    (SpawnPolicy(models=("m",), max_messages=0), _factory, "max_messages"),
    (SpawnPolicy(models=("m",), max_message_chars=0), _factory, "max_message_chars"),
    (SpawnPolicy(models=("m",)), None, "client_factory"),
])
def test_policy_validation(policy, factory, match):
    with pytest.raises(ValueError, match=match):
        validate_spawn_policy(policy, factory)


def test_policy_defaults():
    p = SpawnPolicy(models=("m",))
    assert (p.default_model, p.max_subagents, p.max_depth, p.max_messages,
            p.max_message_chars) == (None, 8, 3, 200, 4000)
    validate_spawn_policy(p, _factory)
    with pytest.raises(TypeError):
        SpawnPolicy(("m",))   # keyword-only


def test_create_agent_without_factory_raises(tmp_path):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(StubChatClient([text("x")]), spawn=POLICY)

    with pytest.raises(ValueError, match="client_factory"):
        asyncio.run(main())


def test_top_node_limits_from_host(tmp_path):
    async def main():
        cfg = SheathConfig(root_dir=tmp_path / "r", subagent_task_timeout_s=50,
                           tool_loop_max_iterations=12)
        async with Session.create(cfg) as sess:
            await sess.create_agent(Rec([text("x")]), spawn=SpawnPolicy(
                models=("m",), max_subagents=5, max_depth=2), client_factory=_factory)
            return sess.delegation.top("data-integrator")

    top = asyncio.run(main())
    assert top.can_spawn
    assert (top.limits.timeout_s, top.limits.max_round_trips, top.limits.subtree_allowance,
            top.limits.remaining_depth) == (50, 12, 5, 2)


# --- refusals: one per row of the spec table, nothing created ------------------------------------

def _refused(tmp_path, call, **kw):
    got: dict = {}
    sess, snap, factory, _ = run_spawn(tmp_path, [call, record(got)], **kw)
    reply = got["sp1"]
    assert reply.startswith("Error: "), reply
    return reply, snap, factory


def test_refuse_name_taken_declared(tmp_path):
    declared = {"lead": SubagentSpec(description="d", client=StubChatClient([text("x")]))}
    reply, snap, factory = _refused(tmp_path, spawn("LEAD", 1), subagents=declared)
    assert "name" in reply and "lead" in reply
    assert snap == {**NOTHING, "nodes": ["lead"], "folders": ["lead"]}
    assert factory.made == []


def test_refuse_name_taken_spawned(tmp_path):
    got: dict = {}
    script = [spawn("a", 1), wait([1], 1), results(1, 1), spawn("A", 2), record(got)]
    _, snap, factory, _ = run_spawn(tmp_path, script)
    assert got["sp2"].startswith("Error: ") and "name" in got["sp2"]
    assert snap == {"nodes": ["a"], "records": 1, "total": 1, "folders": ["a"]}
    assert len(factory.made) == 1


def test_refuse_name_of_top_agent(tmp_path):
    reply, snap, factory = _refused(tmp_path, spawn("Data-Integrator", 1))
    assert snap == NOTHING and factory.made == []


@pytest.mark.parametrize("bad", ["1abc", "has space", "a/b", "../x", "", "x" * 65])
def test_refuse_bad_name(tmp_path, bad):
    reply, snap, factory = _refused(tmp_path, spawn(bad, 1))
    assert "^[A-Za-z][A-Za-z0-9_-]{0,63}$" in reply
    assert snap == NOTHING and factory.made == []


@pytest.mark.parametrize("field", ["instructions", "task"])
def test_refuse_empty_instructions_or_task(tmp_path, field):
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, **{field: "  "}))
    assert field in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_model_not_allowed(tmp_path):
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, model="m-huge"))
    assert "m-huge" in reply and "m-small" in reply and "m-big" in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_unknown_tool(tmp_path):
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, tools=["tool_a", "nope"]),
                                    tools=[tool_a, tool_b])
    assert "nope" in reply and "tool_a" in reply and "tool_b" in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_unknown_tool_lists_none(tmp_path):
    reply, snap, _ = _refused(tmp_path, spawn("a", 1, tools=["read_file"]))
    assert "read_file" in reply and "none" in reply
    assert snap == NOTHING


@pytest.mark.parametrize("asked", [["code"], ["files", "web"], ["bogus"]])
def test_refuse_bundle_not_held(tmp_path, asked):
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, bundles=asked))
    assert asked[-1] in reply and "files" in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_empty_bundles(tmp_path):
    # An empty selection means "every bundle" to the bundle layer: never let it widen.
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, bundles=[]))
    assert "bundle" in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_bad_message_to_name(tmp_path):
    # The top agent may grant any name, but each must be a valid agent name.
    reply, snap, factory = _refused(tmp_path, spawn("a", 1, message_to=["not a name!"]))
    assert "message_to" in reply
    assert snap == NOTHING and factory.made == []


def test_refuse_total_cap(tmp_path):
    got: dict = {}
    policy = SpawnPolicy(models=("m",), max_subagents=1)
    script = [spawn("a", 1), wait([1], 1), results(1, 1), spawn("b", 2), record(got)]
    _, snap, factory, _ = run_spawn(tmp_path, script, spawn_policy=policy)
    assert got["sp2"].startswith("Error: ") and "1 of 1" in got["sp2"]
    assert snap == {"nodes": ["a"], "records": 1, "total": 1, "folders": ["a"]}
    assert len(factory.made) == 1


def test_total_cap_checked_last(tmp_path):
    got: dict = {}
    policy = SpawnPolicy(models=("m",), max_subagents=1)
    script = [spawn("a", 1), wait([1], 1), results(1, 1), spawn("b", 2, model="zzz"),
              record(got)]
    run_spawn(tmp_path, script, spawn_policy=policy)
    assert "zzz" in got["sp2"] and "1 of 1" not in got["sp2"]


def test_parallel_spawns_respect_cap_and_names(tmp_path):
    got: dict = {}
    policy = SpawnPolicy(models=("m",), max_subagents=2)
    script = [[spawn("a", 1), spawn("A", 2), spawn("b", 3), spawn("c", 4)],
              wait([1, 2], 1), wait([2], 2), results(1, 1), results(2, 2), record(got)]
    _, snap, _, _ = run_spawn(tmp_path, script, spawn_policy=policy)
    replies = [got[f"sp{i}"] for i in range(1, 5)]
    assert sum(r.startswith("Background task") for r in replies) == 2, replies
    assert snap["total"] == 2 and snap["nodes"] == ["a", "b"]


# --- clients -----------------------------------------------------------------------------------

def test_refuse_factory_returning_parent_client(tmp_path):
    got: dict = {}
    parent = Rec([spawn("a", 1), record(got)])
    _, snap, _, _ = run_spawn(tmp_path, [], factory=lambda m: parent, parent_client=parent)
    assert got["sp1"].startswith("Error: ") and "client" in got["sp1"]
    assert snap == NOTHING


def test_refuse_factory_returning_shared_client(tmp_path):
    shared = Rec([text("x")])
    got: dict = {}
    script = [spawn("a", 1), wait([1], 1), results(1, 1), spawn("b", 2), record(got)]
    _, snap, _, _ = run_spawn(tmp_path, script, factory=lambda m: shared)
    assert got["sp2"].startswith("Error: ") and "client" in got["sp2"]
    assert snap["nodes"] == ["a"] and snap["total"] == 1


def test_refuse_factory_returning_declared_client(tmp_path):
    declared_client = StubChatClient([text("x")])
    declared = {"lead": SubagentSpec(description="d", client=declared_client)}
    reply, snap, _ = _refused(tmp_path, spawn("a", 1), subagents=declared,
                              factory=lambda m: declared_client)
    assert "client" in reply
    assert snap == {**NOTHING, "nodes": ["lead"], "folders": ["lead"]}


# --- failures after the reservation release it ---------------------------------------------------

def _retry_then(got, before=None):
    """A model step that runs ``before`` (e.g. to clear a fault), then spawns ``a`` again."""
    def step(messages):
        if before is not None:
            before()
        return spawn("a", 2)
    return step


def _assert_released(sess, snap, got):
    assert got["sp1"].startswith("Error: "), got["sp1"]
    assert got["sp2"] == "Background task 1 started on agent 'a'.", got["sp2"]
    assert snap == {"nodes": ["a"], "records": 1, "total": 1, "folders": ["a"]}
    assert sess.delegation._pending == set()


def _retry_script(got, before=None):
    first: dict = {}

    def check_first(messages):
        record(first)(messages)
        got["sp1"] = first["sp1"]
        return _retry_then(got, before)(messages)

    return [spawn("a", 1), check_first, wait([1], 1), results(1, 1), record(got)]


def test_factory_raising_releases_reservation(tmp_path):
    got: dict = {}
    calls = []

    def factory(model):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("secret detail")
        return Rec([text("ok")])

    sess, snap, _, _ = run_spawn(tmp_path, _retry_script(got), factory=factory)
    assert "secret detail" not in got["sp1"]
    _assert_released(sess, snap, got)


def test_create_agent_raising_releases_reservation(tmp_path, monkeypatch):
    got: dict = {}
    original = Session.create_agent
    failed = []

    async def flaky(self, client, **kw):
        if kw.get("name") == "a" and not failed:
            failed.append(1)
            raise RuntimeError("boom")
        return await original(self, client, **kw)

    monkeypatch.setattr(Session, "create_agent", flaky)
    sess, snap, _, _ = run_spawn(tmp_path, _retry_script(got))
    assert failed == [1]
    _assert_released(sess, snap, got)


def test_folder_failure_releases_reservation(tmp_path):
    got: dict = {}
    squatter = tmp_path / "r" / "agents" / "a"
    squatter.parent.mkdir(parents=True)
    squatter.write_text("not a folder")
    seen = []

    def clear():
        seen.append(squatter.is_file())   # the refused spawn left the file alone
        squatter.unlink()

    sess, snap, _, _ = run_spawn(tmp_path, _retry_script(got, clear))
    assert seen == [True]
    _assert_released(sess, snap, got)


def test_round_trips_fail_closed(tmp_path):
    class NoSettings(Rec):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.function_invocation_configuration = None

    reply, snap, _ = _refused(tmp_path, spawn("a", 1), factory=lambda m: NoSettings([text("x")]))
    assert snap == NOTHING


def test_pending_client_counts_for_identity(tmp_path, monkeypatch):
    """Two parallel spawns whose builds interleave must not share one client."""
    original = Session.create_agent

    async def yielding(self, client, **kw):
        if kw.get("name") in ("a", "b"):
            await asyncio.sleep(0.01)
        return await original(self, client, **kw)

    monkeypatch.setattr(Session, "create_agent", yielding)
    shared = Rec([text("x")])
    got: dict = {}
    script = [[spawn("a", 1), spawn("b", 2)], wait([1], 1), results(1, 1), record(got)]
    sess, snap, _, _ = run_spawn(tmp_path, script, factory=lambda m: shared)
    replies = sorted([got["sp1"], got["sp2"]])
    assert replies[0].startswith("Background task 1") and replies[1].startswith("Error: ")
    assert "client" in replies[1]
    assert snap["total"] == 1 and sess.delegation._pending_clients == []


def test_second_policy_on_session_refused(tmp_path):
    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(Rec([text("x")]), name="one", spawn=POLICY,
                                    client_factory=_factory)
            await sess.create_agent(Rec([text("x")]), name="two", spawn=POLICY,
                                    client_factory=_factory)   # same policy object: fine
            with pytest.raises(ValueError, match="SpawnPolicy"):
                await sess.create_agent(Rec([text("x")]), name="three",
                                        spawn=SpawnPolicy(models=("m",)),
                                        client_factory=_factory)

    asyncio.run(main())
