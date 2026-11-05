"""Runtime sub-agents that spawn their own: nesting, per-sub-agent limits and the cascade."""

from __future__ import annotations

import asyncio

import pytest

from agent_framework import ChatResponse, Message

from sheath import SheathConfig, Session
from sheath.subagents import SpawnPolicy, SubagentSpec
from sheath.testing import text, tool_call

from tests.test_spawn import NOTHING, Rec, snapshot, spawn
from tests.test_subagents import record, results, start, wait

# --- helpers -----------------------------------------------------------------------------------


class Queue:
    """A client factory handing out scripts in creation order (one per spawned sub-agent)."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.made: list[tuple[str, Rec]] = []

    def __call__(self, model):
        script = self.scripts.pop(0) if self.scripts else [text("spare")]
        client = script if isinstance(script, Rec) else Rec(script, model=model)
        self.made.append((model, client))
        return client


def offered(client) -> set[str]:
    """Every tool name the client was offered over all its model calls."""
    return set().union(*client.calls) if client.calls else set()


def hold(*names):
    """A model step that blocks until the tasks on every agent in ``names`` have ended, then
    calls ``probe`` (a no-op tool) so the next model call carries any queued notices."""
    def step(messages):
        return tool_call("probe", {"names": list(names)}, f"probe-{'-'.join(names)}")
    return step


def tools_for(delegation_box):
    """``probe``: wait (bounded) until every named agent's tasks have ended."""

    async def probe(names: list[str]) -> str:
        """Wait for the named agents' tasks to end."""
        d = delegation_box["d"]

        async def ended():
            while True:
                recs = [r for r in d._records if r.agent.name in names]
                if len({r.agent.name for r in recs}) == len(names) and all(
                        r.future is None or r.future.done() for r in recs):
                    return
                await asyncio.sleep(0.005)

        await asyncio.wait_for(ended(), 5)
        await asyncio.sleep(0)   # let the done callbacks deliver their notices
        return "ended"

    return probe


def run_tree(tmp_path, script, factory, *, policy=None, tools=None, bundles=("files",),
             cfg=None, events=None, subagents=None, parent_client=None, box=None):
    config = SheathConfig(root_dir=tmp_path / "r", **(cfg or {}))
    parent = parent_client or Rec(script)
    policy = policy or SpawnPolicy(models=("m",))

    async def main():
        async with Session.create(config) as sess:
            if events is not None:
                sess.subscribe(events.append)
            if box is not None:
                box["d"] = sess.delegation
                box["sess"] = sess
            agent = await sess.create_agent(parent, tools=tools, bundles=bundles,
                                            subagents=subagents, spawn=policy,
                                            client_factory=factory)
            await agent.run("go", session=agent.create_session())
            return sess, snapshot(sess)

    async def guarded():
        return await asyncio.wait_for(main(), 5)

    sess, snap = asyncio.run(guarded())
    return sess, snap, parent


def tool_a(q: str) -> str:
    """Host tool A."""
    return f"a:{q}"


def tool_b(q: str) -> str:
    """Host tool B."""
    return f"b:{q}"


HAND = ["spawn_subagent"]


# --- nesting -----------------------------------------------------------------------------------

def test_spawn_tool_handed_down_enables_nesting(tmp_path):
    events: list = []
    got: dict = {}
    lead_got: dict = {}
    lead = [spawn("child", 1, tools=["tool_a"]), wait([2], 1), results(2, 1), record(lead_got,
            lambda m: text("lead-report"))]
    child = [tool_call("tool_a", {"q": "x"}, "a1"), text("child-report")]
    factory = Queue(lead, child)
    script = [spawn("lead", 1, tools=["tool_a", "tool_b", *HAND]), wait([1], 1), results(1, 1),
              record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory, tools=[tool_a, tool_b], events=events)
    d = sess.delegation
    lead_node, child_node = d.node("lead"), d.node("child")
    assert got["r1"] == "lead-report"
    assert lead_got["r1"] == "child-report"
    assert lead_node.can_spawn and not child_node.can_spawn
    assert child_node.path == "lead/child" and child_node.depth == 2
    assert child_node.creator is lead_node
    # The child's tools are a subset of the lead's, as the same objects.
    assert set(child_node.tool_objects) <= set(lead_node.tool_objects)
    assert child_node.tool_objects["tool_a"] is lead_node.tool_objects["tool_a"]
    lead_client, child_client = factory.made[0][1], factory.made[1][1]
    assert "spawn_subagent" in offered(lead_client)
    assert "background_agents_wait_for_first_completion" in offered(lead_client)
    assert offered(child_client) - {"tool_a"} <= offered(lead_client)
    assert "spawn_subagent" not in offered(child_client)
    # Folders follow the path.
    assert child_node.folder == "agents/lead/child"
    assert (sess.root / "agents" / "lead" / "child").is_dir()
    assert snap["total"] == 2 and snap["nodes"] == ["child", "lead"]
    # Status is tagged with the node path everywhere.
    sheath_events = [(e.message, e.agent) for e in events if e.tool == "sheath"]
    assert ("started", "lead/child") in sheath_events
    assert ("finished", "lead/child") in sheath_events
    activity = [(e.message, e.agent) for e in events if e.tool == "tool_a"]
    assert activity == [("running", "lead/child"), ("done", "lead/child")]


def test_not_handed_down_not_offered(tmp_path):
    got: dict = {}
    factory = Queue([text("lead-report")])
    script = [spawn("lead", 1, tools=["tool_a"]), wait([1], 1), results(1, 1), record(got)]
    sess, _, _ = run_tree(tmp_path, script, factory, tools=[tool_a])
    lead_client = factory.made[0][1]
    assert lead_client.calls
    assert not any(n == "spawn_subagent" or n.startswith("background_agents")
                   for n in offered(lead_client))
    assert not sess.delegation.node("lead").can_spawn


@pytest.mark.parametrize("max_depth, refused_at", [(1, "lead"), (2, "child")])
def test_max_depth_refuses_hand_down(tmp_path, max_depth, refused_at):
    got: dict = {}
    lead_got: dict = {}
    lead = [spawn("child", 1, tools=HAND), record(lead_got,
            lambda m: wait([2], 2) if not lead_got["sp1"].startswith("Error") else text("x")),
            results(2, 2), text("lead-report")]
    factory = Queue(lead, [text("child-report")])
    script = [spawn("lead", 1, tools=HAND), record(got, lambda m: wait([1], 1)),
              results(1, 1), text("done")]
    sess, snap, _ = run_tree(tmp_path, script, factory,
                             policy=SpawnPolicy(models=("m",), max_depth=max_depth))
    reply = got["sp1"] if refused_at == "lead" else lead_got["sp1"]
    assert reply.startswith("Error: ") and "max_depth" in reply and "spawn_subagent" in reply
    if refused_at == "lead":
        assert snap == NOTHING and factory.made == []
    else:
        assert snap["nodes"] == ["lead"] and snap["total"] == 1


# --- limits ------------------------------------------------------------------------------------

HOST = {"subagent_task_timeout_s": 100, "tool_loop_max_iterations": 10}


@pytest.mark.parametrize("kw, arg", [
    ({"timeout_s": 101}, "timeout_s"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"timeout_s": -1}, "timeout_s"),
    ({"max_round_trips": 11}, "max_round_trips"),
    ({"max_round_trips": 0}, "max_round_trips"),
    ({"max_subagents": 9, "tools": HAND}, "max_subagents"),
    ({"max_subagents": 0, "tools": HAND}, "max_subagents"),
    ({"max_depth": 3, "tools": HAND}, "max_depth"),
    ({"max_depth": 0, "tools": HAND}, "max_depth"),
])
def test_limits_tighten_only(tmp_path, kw, arg):
    got: dict = {}
    factory = Queue()
    _, snap, _ = run_tree(tmp_path, [spawn("lead", 1, **kw), record(got)], factory, cfg=HOST)
    reply = got["sp1"]
    assert reply.startswith("Error: ") and arg in reply, reply
    assert snap == NOTHING and factory.made == []


def test_limits_tighten_only_nested(tmp_path):
    got: dict = {}
    lead_got: dict = {}
    lead = [spawn("child", 1, timeout_s=51), record(lead_got, lambda m: text("lead-report"))]
    factory = Queue(lead)
    script = [spawn("lead", 1, tools=HAND, timeout_s=50), wait([1], 1), results(1, 1),
              record(got)]
    _, snap, _ = run_tree(tmp_path, script, factory, cfg=HOST)
    assert lead_got["sp1"].startswith("Error: ") and "timeout_s" in lead_got["sp1"]
    assert "50" in lead_got["sp1"]
    assert snap["nodes"] == ["lead"] and len(factory.made) == 1


@pytest.mark.parametrize("kw", [{"max_subagents": 1}, {"max_depth": 1}])
def test_limit_args_require_spawn_tool(tmp_path, kw):
    got: dict = {}
    factory = Queue()
    _, snap, _ = run_tree(tmp_path, [spawn("lead", 1, **kw), record(got)], factory)
    reply = got["sp1"]
    assert reply.startswith("Error: ") and next(iter(kw)) in reply
    assert "spawn_subagent" in reply
    assert snap == NOTHING and factory.made == []


def test_timeout_and_round_trips_without_spawn_tool_allowed(tmp_path):
    got: dict = {}
    factory = Queue([text("ok")])
    script = [spawn("lead", 1, timeout_s=5, max_round_trips=3), wait([1], 1), results(1, 1),
              record(got)]
    sess, _, _ = run_tree(tmp_path, script, factory, cfg=HOST)
    assert got["sp1"].startswith("Background task 1")
    limits = sess.delegation.node("lead").limits
    assert (limits.timeout_s, limits.max_round_trips) == (5, 3)


def _limits(node):
    lim = node.limits
    return (lim.timeout_s, lim.max_round_trips, lim.subtree_allowance, lim.remaining_depth)


def test_omitted_limits_inherit(tmp_path):
    lead = [spawn("child", 1, tools=HAND), wait([2], 1), results(2, 1), text("lead-report")]
    factory = Queue(lead, [text("child-report")])
    script = [spawn("lead", 1, tools=HAND), wait([1], 1), results(1, 1), text("done")]
    sess, _, _ = run_tree(tmp_path, script, factory, cfg=HOST,
                          policy=SpawnPolicy(models=("m",), max_subagents=8, max_depth=3))
    d = sess.delegation
    # The lead: the host's timeout and round trips; the remaining conversation budget after
    # itself; one level less than the top agent.
    assert _limits(d.node("lead")) == (100, 10, 7, 2)
    # The child: the lead's, its allowance what remained of the lead's after itself.
    assert _limits(d.node("child")) == (100, 10, 6, 1)


def test_explicit_limits_fixed_at_creation(tmp_path):
    factory = Queue([text("lead-report")])
    script = [spawn("lead", 1, tools=HAND, timeout_s=30, max_round_trips=4, max_subagents=3,
                    max_depth=1), wait([1], 1), results(1, 1), text("done")]
    sess, _, parent = run_tree(tmp_path, script, factory, cfg=HOST)
    assert _limits(sess.delegation.node("lead")) == (30, 4, 3, 1)
    desc = factory.made[0][1].calls[0]["spawn_subagent"]
    assert "3" in desc and "max_subagents" in desc and "max_depth" in desc


def test_subtree_allowance(tmp_path):
    got: dict = {}
    lead_got: dict = {}
    lead2_got: dict = {}
    lead = [spawn("c1", 1), spawn("c2", 2), spawn("c3", 3), wait([2, 3], 1), results(2, 1),
            results(3, 2), record(lead_got, lambda m: text("lead-report"))]
    lead2 = [spawn("d1", 1), wait([5], 1), results(5, 1),
             record(lead2_got, lambda m: text("lead2-report"))]
    factory = Queue(lead, [text("c1")], [text("c2")], lead2, [text("d1")])
    script = [spawn("lead", 1, tools=HAND, max_subagents=2), wait([1], 1), results(1, 1),
              spawn("lead2", 2, tools=HAND), wait([4], 2), results(4, 2), record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory)
    assert lead_got["sp1"].startswith("Background task 2")
    assert lead_got["sp2"].startswith("Background task 3")
    third = lead_got["sp3"]
    assert third.startswith("Error: ") and "max_subagents" in third and "2 of 2" in third
    assert "lead" in third
    # A sibling still spawns from the conversation budget.
    assert lead2_got["sp1"].startswith("Background task 5")
    assert snap["total"] == 5 and "c3" not in snap["nodes"]


def test_grandchild_counts_against_lead_allowance(tmp_path):
    lead_got: dict = {}
    c1_got: dict = {}
    lead = [spawn("c1", 1, tools=HAND), wait([2], 1), results(2, 1), spawn("c2", 2),
            record(lead_got, lambda m: text("lead-report"))]
    c1 = [spawn("g1", 1), wait([3], 1), results(3, 1), spawn("g2", 2),
          record(c1_got, lambda m: text("c1-report"))]
    factory = Queue(lead, c1, [text("g1")])
    script = [spawn("lead", 1, tools=HAND, max_subagents=2), wait([1], 1), results(1, 1),
              text("done")]
    sess, snap, _ = run_tree(tmp_path, script, factory)
    assert c1_got["sp1"].startswith("Background task 3")
    assert c1_got["sp2"].startswith("Error: ") and "max_subagents" in c1_got["sp2"]
    assert lead_got["sp2"].startswith("Error: ") and "max_subagents" in lead_got["sp2"]
    assert "2 of 2" in lead_got["sp2"]
    assert snap["nodes"] == ["c1", "g1", "lead"] and snap["total"] == 3
    assert sess.delegation.node("g1").path == "lead/c1/g1"


def test_timeout_applies_to_later_continue(tmp_path):
    got: dict = {}

    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(3)
        return "late"

    factory = Queue([text("quick"), tool_call("slow", call_id="s"), text("never")])
    script = [spawn("lead", 1, tools=["slow"], timeout_s=0.1), wait([1], 1), results(1, 1),
              tool_call("background_agents_continue_task", {"task_id": 1, "text": "more"}, "c1"),
              wait([1], 2), results(1, 2), record(got)]
    run_tree(tmp_path, script, factory, tools=[slow], cfg={"subagent_task_timeout_s": None})
    assert got["r1"] == "quick"
    assert got["r2"].startswith("Task failed") and "timeout_s" in got["r2"]


def test_round_trips_on_own_client(tmp_path):
    factory = Queue([text("a")], [text("b")])
    script = [spawn("lead", 1, max_round_trips=4), spawn("other", 2), wait([1, 2], 1),
              wait([1, 2], 2), results(1, 1), results(2, 2), text("done")]
    _, _, parent = run_tree(tmp_path, script, factory, cfg=HOST)
    lead_client, other_client = factory.made[0][1], factory.made[1][1]
    assert lead_client.function_invocation_configuration["max_iterations"] == 4
    assert other_client.function_invocation_configuration["max_iterations"] == 10
    assert parent.function_invocation_configuration["max_iterations"] == 10


# --- the cascade --------------------------------------------------------------------------------

def test_cascade_cancels_children(tmp_path):
    got: dict = {}
    state: dict = {"cancelled": [], "entered": None}

    async def block(who: str) -> str:
        """Block until cancelled."""
        state["entered"].set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"].append(who)
            raise
        return "unblocked"

    async def slow() -> str:
        """Work past the deadline once the grandchild is blocked."""
        await state["entered"].wait()
        await asyncio.sleep(30)
        return "late"

    lead = [spawn("child", 1, tools=["block", *HAND]), tool_call("slow", call_id="s"),
            text("never")]
    child = [spawn("grand", 1, tools=["block"]), wait([3], 1), results(3, 1), text("never")]
    grand = [tool_call("block", {"who": "grand"}, "b"), text("never")]
    factory = Queue(lead, child, grand)
    box: dict = {}

    def after_lead(messages):
        record(got)(messages)
        d = box["d"]
        box["at_read"] = {r.agent.name: (r.status, r.future is None or r.future.done())
                          for r in d._records}
        return text("done")

    script = [spawn("lead", 1, tools=["block", "slow", *HAND], timeout_s=0.5), wait([1], 1),
              results(1, 1), after_lead]

    async def entered_event():
        state["entered"] = asyncio.Event()

    config = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=4)

    async def main():
        await entered_event()
        async with Session.create(config) as sess:
            box["d"] = sess.delegation
            agent = await sess.create_agent(Rec(script), tools=[block, slow], bundles=("files",),
                                            spawn=SpawnPolicy(models=("m",)),
                                            client_factory=factory)
            await agent.run("go", session=agent.create_session())
            return sess

    sess = asyncio.run(asyncio.wait_for(main(), 5))
    assert "timeout_s" in got["r1"]
    assert state["cancelled"] == ["grand"]
    # By the time the lead's failure was read, its child and grandchild had ended too.
    assert box["at_read"]["child"][1] and box["at_read"]["grand"][1]
    d = sess.delegation
    for r in d._records:
        assert r.status == "failed", (r.agent.name, r.status)
        assert not r.unread or r.agent.name == "lead"


def test_nested_no_deadlock(tmp_path):
    a_got: dict = {}
    b_got: dict = {}
    lead_a = [spawn("ca", 1), wait([3], 1), results(3, 1),
              record(a_got, lambda m: text("a-report"))]
    lead_b = [spawn("cb", 1), wait([4], 1), results(4, 1),
              record(b_got, lambda m: text("b-report"))]
    got: dict = {}
    factory = Queue(lead_a, lead_b, [text("ca-report")], [text("cb-report")])
    script = [[spawn("leadA", 1, tools=HAND), spawn("leadB", 2, tools=HAND)],
              wait([1, 2], 1), wait([1, 2], 2), results(1, 1), results(2, 2), record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory, cfg={"subagent_max_concurrent": 2})
    assert got["r1"] == "a-report" and got["r2"] == "b-report"
    assert a_got["r1"] == "ca-report" and b_got["r1"] == "cb-report"
    assert snap["total"] == 4


# --- declared roles ------------------------------------------------------------------------------

def test_declared_role_can_spawn(tmp_path):
    events: list = []
    got: dict = {}
    role_got: dict = {}
    role_client = Rec([spawn("helper", 1), wait([2], 1), results(2, 1),
                       record(role_got, lambda m: text("role-report"))])
    plain_client = Rec([text("plain-report")])
    subs = {"role": SubagentSpec(description="d", client=role_client, can_spawn=True),
            "plain": SubagentSpec(description="d", client=plain_client)}
    factory = Queue([text("helper-report")])
    script = [start("role", 1), wait([1], 1), results(1, 1), start("plain", 2), wait([3], 2),
              results(3, 2), record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory, subagents=subs, events=events,
                             cfg=HOST)
    assert role_got["r1"] == "helper-report" and got["r1"] == "role-report"
    assert "spawn_subagent" in offered(role_client)
    assert "spawn_subagent" not in offered(plain_client)
    d = sess.delegation
    assert d.node("helper").path == "role/helper" and d.node("helper").creator is d.node("role")
    assert ("started", "role/helper") in [(e.message, e.agent) for e in events
                                           if e.tool == "sheath"]
    assert snap["total"] == 1
    assert SubagentSpec(description="d").can_spawn is False


def test_declared_can_spawn_without_policy_offers_nothing(tmp_path):
    role_client = Rec([text("role-report")])
    subs = {"role": SubagentSpec(description="d", client=role_client, can_spawn=True)}

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            agent = await sess.create_agent(Rec([start("role", 1), wait([1], 1), results(1, 1),
                                                 text("done")]),
                                            bundles=("files",), subagents=subs)
            await agent.run("go", session=agent.create_session())

    asyncio.run(asyncio.wait_for(main(), 5))
    assert role_client.calls and "spawn_subagent" not in offered(role_client)


# --- two concurrent runs on one node --------------------------------------------------------------

class PerRun(Rec):
    """One client serving several concurrent runs: each run (told apart by its task input)
    follows its own script, and its model calls are kept apart for inspection."""

    def __init__(self, scripts):
        super().__init__([])
        self.scripts = scripts
        self.turns = {k: 0 for k in scripts}
        self.seen: dict[str, list[list[str]]] = {k: [] for k in scripts}

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls.append({t.name: (t.description or "") for t in options.get("tools") or []})
        texts = [m.text for m in messages]
        key = next(k for k in self.scripts if k in texts)
        self.seen[key].append(texts)
        script = self.scripts[key]
        step = script[min(self.turns[key], len(script) - 1)]
        self.turns[key] += 1
        if callable(step):
            step = step(list(messages))

        async def _response():
            return ChatResponse(messages=Message(role="assistant", contents=[step]))
        return _response()


def _own_task(name):
    """Read the result of the task on ``name`` (its id found in this run's tool results)."""
    def step(messages):
        for m in messages:
            for c in m.contents:
                if c.type == "function_result" and f"agent '{name}'" in str(c.result):
                    tid = int(str(c.result).split()[2])
                    return results(tid, f"own-{name}")
        raise AssertionError(f"no task on {name}")
    return step


def wait_on(name):
    """Wait for the task on ``name`` (its id found in this run's tool results)."""
    def step(messages):
        tid_step = _own_task(name)(messages)
        tid = tid_step.arguments["task_id"]
        return wait([tid], f"on-{name}")
    return step


def _own_task_of_input(_):
    """Read the second task on ``lead`` (the one started by name, after the spawn's)."""
    def step(messages):
        for m in messages:
            for c in m.contents:
                if c.type == "function_result" and str(c.result).startswith("Background task") \
                        and c.call_id == "s2":
                    return results(int(str(c.result).split()[2]), "own-b")
        raise AssertionError("no second task on lead")
    return step


def test_two_concurrent_runs_on_one_node_each_get_own_notice(tmp_path):
    box: dict = {}
    probe = tools_for(box)
    lead_client = PerRun({
        "run-A": [spawn("gA", 1), hold("gA", "gB"), _own_task("gA"), text("A-report")],
        "run-B": [spawn("gB", 1), hold("gA", "gB"), _own_task("gB"), text("B-report")],
    })
    factory = Queue(lead_client, [text("gA-report")], [text("gB-report")])
    got: dict = {}
    script = [spawn("lead", 1, tools=["probe", *HAND], task="run-A"),
              tool_call("background_agents_start_task",
                        {"agent_name": "lead", "input": "run-B", "description": "b"}, "s2"),
              hold("lead"), _own_task("lead"), _own_task_of_input("run-B"), record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory, tools=[probe], box=box,
                             cfg={"subagent_max_concurrent": 4})
    assert got["rown-lead"] == "A-report" and got["rown-b"] == "B-report"
    d = sess.delegation
    ids = {r.agent.name: r.id for r in d._records}

    def notice(name):
        return f"[task {ids[name]} (lead/{name}) completed"

    def got_notice(run, name):
        return any(notice(name) in t for call in lead_client.seen[run] for t in call if t)

    assert got_notice("run-A", "gA") and got_notice("run-B", "gB")
    assert not got_notice("run-A", "gB") and not got_notice("run-B", "gA")
    assert d.node("lead").inbox == []


def test_concurrent_run_waits_for_and_cancels_only_its_own_tasks(tmp_path):
    """Run B of the lead ends while run A's sub-agent still works: B's parent loop does not wait
    for A's task, and B's end does not cancel it."""
    box: dict = {}
    state: dict = {}

    async def gate() -> str:
        """Block until the host opens the gate."""
        await state["open"].wait()
        return "opened"

    async def await_b() -> str:
        """Wait (bounded) for run B of the lead to end, then open the gate."""
        d = box["d"]

        async def b_done():
            while True:
                recs = [r for r in d._records if r.description == "b"]
                if recs and (recs[0].future is None or recs[0].future.done()):
                    return
                await asyncio.sleep(0.005)

        try:
            await asyncio.wait_for(b_done(), 2)
            outcome = "b-ended-first"
        except TimeoutError:
            outcome = "b-still-running"
        state["open"].set()
        return outcome

    lead_client = PerRun({
        "run-A": [spawn("gA", 1, tools=["gate"]), wait_on("gA"), _own_task("gA"),
                  text("A-report")],
        "run-B": [spawn("gB", 1), wait_on("gB"), _own_task("gB"), text("B-report")],
    })
    factory = Queue(lead_client, [tool_call("gate", call_id="g"), text("gA-report")],
                    [text("gB-report")])
    got: dict = {}
    script = [spawn("lead", 1, tools=["gate", *HAND], task="run-A"),
              tool_call("background_agents_start_task",
                        {"agent_name": "lead", "input": "run-B", "description": "b"}, "s2"),
              tool_call("await_b", call_id="ab"), hold("lead"), _own_task("lead"),
              _own_task_of_input("run-B"), record(got)]

    probe = tools_for(box)

    async def main():
        state["open"] = asyncio.Event()
        config = SheathConfig(root_dir=tmp_path / "r", subagent_max_concurrent=4)
        async with Session.create(config) as sess:
            box["d"] = sess.delegation
            agent = await sess.create_agent(Rec(script), tools=[gate, await_b, probe],
                                            bundles=("files",),
                                            spawn=SpawnPolicy(models=("m",)),
                                            client_factory=factory)
            await agent.run("go", session=agent.create_session())
            return sess.delegation

    d = asyncio.run(asyncio.wait_for(main(), 5))
    assert got["ab"] == "b-ended-first"
    by_agent = {r.agent.name: r for r in d._records}
    assert by_agent["gA"].status == "completed"            # run B's end left it running
    assert by_agent["gA"].result_text == "gA-report"
    assert got["rown-lead"] == "A-report" and got["rown-b"] == "B-report"
    # Run B was never nudged about run A's task.
    assert not any("lead/gA" in t and "must finish" in t
                   for call in lead_client.seen["run-B"] for t in call if t)


# --- non-finite timeouts (NaN slips past <= and > comparisons; inf is no bound) -------------------

@pytest.mark.parametrize("value", [float("nan"), "nan", float("inf"), "inf", float("-inf")])
def test_non_finite_timeout_refused(tmp_path, value):
    got: dict = {}
    factory = Queue()
    _, snap, _ = run_tree(tmp_path, [spawn("lead", 1, timeout_s=value), record(got)], factory,
                          cfg=HOST)
    reply = got["sp1"]
    assert reply.startswith("Error: ") and "timeout_s" in reply, reply
    assert snap == NOTHING and factory.made == []


@pytest.mark.parametrize("value", [float("inf")])
def test_non_finite_timeout_refused_unbounded_host(tmp_path, value):
    got: dict = {}
    factory = Queue()
    _, snap, _ = run_tree(tmp_path, [spawn("lead", 1, timeout_s=value), record(got)], factory,
                          cfg={"subagent_task_timeout_s": None})
    assert got["sp1"].startswith("Error: ") and "timeout_s" in got["sp1"], got["sp1"]
    assert snap == NOTHING


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e9])
def test_grandchild_never_exceeds_host_timeout(tmp_path, value):
    got: dict = {}
    lead_got: dict = {}
    lead = [spawn("child", 1, timeout_s=value), record(lead_got, lambda m: text("lead-report"))]
    factory = Queue(lead)
    script = [spawn("lead", 1, tools=HAND), wait([1], 1), results(1, 1), record(got)]
    sess, snap, _ = run_tree(tmp_path, script, factory, cfg=HOST)
    assert lead_got["sp1"].startswith("Error: ") and "timeout_s" in lead_got["sp1"]
    assert snap["nodes"] == ["lead"]
    assert all(n.limits.timeout_s is not None and n.limits.timeout_s <= 100
               for n in sess.delegation._nodes.values() if n.depth > 0)


def test_effective_limits_refuses_non_finite_directly():
    import math

    from sheath.delegation import AgentNode, Limits, effective_limits

    node = AgentNode(name="t", path="t", depth=0,
                     limits=Limits(timeout_s=100, max_round_trips=10, subtree_allowance=5,
                                   remaining_depth=2))
    for value in (math.nan, math.inf, -math.inf):
        out = effective_limits(node, {"timeout_s": value})
        assert isinstance(out, str) and out.startswith("Error: ") and "timeout_s" in out


def test_effective_limits_refuses_int_too_large_for_float():
    from sheath.delegation import AgentNode, Limits, effective_limits

    for timeout in (100, None):
        node = AgentNode(name="t", path="t", depth=0,
                         limits=Limits(timeout_s=timeout, max_round_trips=10,
                                       subtree_allowance=5, remaining_depth=2))
        out = effective_limits(node, {"timeout_s": 10 ** 400})
        assert isinstance(out, str) and out.startswith("Error: ") and "timeout_s" in out
