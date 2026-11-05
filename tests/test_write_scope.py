"""Sub-agent write confinement: its own folder, plus the paths its creator granted."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from sheath import SheathConfig, Session
from sheath.bundles import subagent_instructions
from sheath.delegation import CURRENT_RUN, AgentNode, RunState
from sheath.subagents import SubagentSpec
from sheath.testing import text, tool_call
from sheath.tools.files import edit_file, read_file, write_file

from tests.test_spawn import Rec, spawn
from tests.test_spawn_nesting import Queue, run_tree
from tests.test_subagents import record, results, start, wait


@contextmanager
def as_agent(node):
    token = CURRENT_RUN.set(RunState(node=node))
    try:
        yield
    finally:
        CURRENT_RUN.reset(token)


def _node(path="a", write_to=()):
    return AgentNode(name=path.rsplit("/", 1)[-1], path=path, depth=path.count("/") + 1,
                     folder=f"agents/{path}", write_to=set(write_to))


@pytest.fixture
def sess(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


# --- enforcement -------------------------------------------------------------------------------

def test_own_folder_writable(sess):
    with as_agent(_node("a")):
        write_file(sess, "agents/a/notes.md", "n")
        write_file(sess, "agents/a/deep/x.md", "x")
    assert (sess.root / "agents/a/deep/x.md").read_text() == "x"


@pytest.mark.parametrize("path", ["x.md", "agents/b/x.md", "agents/ab/x.md", "agents/a",
                                  "agents/a/../b/x.md", "agents/x.md"])
def test_outside_folder_refused(sess, path):
    with as_agent(_node("a")):
        with pytest.raises(ValueError, match="agents/a/") as e:
            write_file(sess, path, "x")
    assert "outside where you may write" in str(e.value)
    assert not (sess.root / "x.md").exists() and not (sess.root / "agents/b").exists()


def test_refused_before_creating_parent_dirs(sess):
    with as_agent(_node("a")):
        with pytest.raises(ValueError):
            write_file(sess, "made/up/dirs/x.md", "x")
    assert not (sess.root / "made").exists()


def test_edit_outside_refused_and_unchanged(sess):
    write_file(sess, "shared.md", "top text")            # the top agent
    with as_agent(_node("a")):
        read_file(sess, "shared.md")                     # reading is allowed anywhere
        with pytest.raises(ValueError, match="outside where you may write"):
            edit_file(sess, "shared.md", "top", "sub")
    assert (sess.root / "shared.md").read_text() == "top text"


def test_creator_may_write_into_its_subagents_folders(sess):
    with as_agent(_node("lead")):
        write_file(sess, "agents/lead/child/brief.md", "b")
    with as_agent(_node("lead/child")):
        write_file(sess, "agents/lead/child/out.md", "o")
        with pytest.raises(ValueError, match="agents/lead/child/"):
            write_file(sess, "agents/lead/x.md", "x")


def test_grants(sess):
    node = _node("a", write_to={"out", "reports/final.md"})
    with as_agent(node):
        write_file(sess, "out/r.md", "r")
        write_file(sess, "out/sub/r.md", "r")
        write_file(sess, "reports/final.md", "f")
        for bad in ("outside/x.md", "reports/final.md.bak", "reports/other.md", "out2/x"):
            with pytest.raises(ValueError, match="agents/a/, out, reports/final.md"):
                write_file(sess, bad, "x")


def test_symlink_out_of_folder_refused(sess):
    (sess.root / "agents/a").mkdir(parents=True)
    (sess.root / "agents/b").mkdir(parents=True)
    (sess.root / "agents/a/link").symlink_to(sess.root / "agents/b")
    with as_agent(_node("a")):
        with pytest.raises(ValueError):
            write_file(sess, "agents/a/link/x.md", "x")
    assert not (sess.root / "agents/b/x.md").exists()


def test_top_agent_unrestricted(sess):
    write_file(sess, "anywhere.md", "t")
    write_file(sess, "agents/a/x.md", "t")
    with as_agent(AgentNode(name="top", path="top", depth=0)):
        write_file(sess, "also.md", "t")


def test_publish_handle_confined(tmp_path):
    host_calls: list = []
    s = Session.create(SheathConfig(root_dir=tmp_path / "r"),
                       on_publish=lambda pf: host_calls.append(pf) or {"ok": 1})
    from sheath.tools.deliver import publish_handle

    h = s.store.put("hello", source="test")
    with as_agent(_node("a")):
        out = publish_handle(s, h.id, name="greeting.txt")
    assert "error" in out and "outputs/greeting.txt" in out["error"]
    assert not host_calls and not (s.root / "outputs/greeting.txt").exists()
    with as_agent(_node("a", write_to={"outputs"})):
        out = publish_handle(s, h.id, name="greeting.txt")
    assert "error" not in out and len(host_calls) == 1


# --- grants through spawn_subagent and SubagentSpec ---------------------------------------------

def _write(path, i):
    return tool_call("write_file", {"path": path, "content": "c"}, f"wf{i}")


def test_spawn_write_to_grant(tmp_path):
    a_got: dict = {}
    a = [_write("out/report.md", 1), _write("stray.md", 2), record(a_got, lambda m: text("ok"))]
    got: dict = {}
    script = [spawn("a", 1, write_to=["out"]), wait([1], 1), results(1, 1), record(got)]
    sess, _, _ = run_tree(tmp_path, script, Queue(a))
    assert got["sp1"] == "Background task 1 started on agent 'a'."
    assert (sess.root / "out/report.md").exists()
    assert not (sess.root / "stray.md").exists()
    assert "Error" in a_got["wf2"] and "agents/a/" in a_got["wf2"]


@pytest.mark.parametrize("grant, ok", [(["out"], False), (["agents/lead/shared"], True),
                                       (["agents/other"], False), (["."], False)])
def test_spawned_grant_must_be_within_creator_scope(tmp_path, grant, ok):
    lead_got: dict = {}
    lead = [spawn("child", 1, write_to=grant), record(lead_got, lambda m: text("lead-done"))]
    child = [text("child-done")]
    script = [spawn("lead", 1, tools=["spawn_subagent"]), wait([1], 1), results(1, 1),
              text("done")]
    run_tree(tmp_path, script, Queue(lead, child))
    reply = lead_got["sp1"]
    if ok:
        assert reply.startswith("Background task")
    else:
        assert reply.startswith("Error: cannot grant write access") and "agents/lead/" in reply


@pytest.mark.parametrize("grant", ["out", [5], [""], ["../escape"], ["/etc"]])
def test_bad_write_to_refused(tmp_path, grant):
    got: dict = {}
    script = [spawn("a", 1, write_to=grant), record(got)]
    sess, _, _ = run_tree(tmp_path, script, Queue([text("a")]))
    assert got["sp1"].startswith("Error: ")
    if isinstance(grant, list) and all(isinstance(g, str) for g in grant):
        assert "write_to" in got["sp1"]    # a mistyped grant fails the tool's argument schema
    assert sess.delegation.node("a") is None


def test_top_may_grant_anywhere(tmp_path):
    got: dict = {}
    script = [spawn("a", 1, write_to=["."]), wait([1], 1), results(1, 1), record(got)]
    a = [_write("root.md", 1), text("a-done")]
    sess, _, _ = run_tree(tmp_path, script, Queue(a))
    assert (sess.root / "root.md").exists()


def test_declared_role_write_to(tmp_path):
    role_got: dict = {}
    role = Rec([_write("reports/r.md", 1), _write("x.md", 2), _write("agents/role/n.md", 3),
                record(role_got, lambda m: text("role-done"))])
    subs = {"role": SubagentSpec(description="d", client=role, write_to=("reports",))}
    script = [start("role", 1), wait([1], 1), results(1, 1), text("done")]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
            agent = await s.create_agent(Rec(script), bundles=("files",), subagents=subs)
            await agent.run("go", session=agent.create_session())
            return s

    s = asyncio.run(asyncio.wait_for(main(), 5))
    assert (s.root / "reports/r.md").exists() and (s.root / "agents/role/n.md").exists()
    assert not (s.root / "x.md").exists()
    assert "agents/role/" in role_got["wf2"]


@pytest.mark.parametrize("bad", ["reports", ("../up",), (5,)])
def test_declared_role_bad_write_to(tmp_path, bad):
    subs = {"role": SubagentSpec(description="d", client=Rec([text("x")]), write_to=bad)}

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
            await s.create_agent(Rec([text("x")]), bundles=("files",), subagents=subs)

    with pytest.raises((ValueError, TypeError), match="write_to"):
        asyncio.run(main())


# --- instructions ------------------------------------------------------------------------------

def test_instructions_state_the_scope():
    plain = subagent_instructions("lead/child")
    assert "only in your folder, agents/lead/child/" in plain
    assert "run_python" in plain
    granted = subagent_instructions("a", ("out", "reports/final.md"))
    assert "out" in granted and "reports/final.md" in granted


class InstrRec(Rec):
    """Records the instructions and system text of each model call."""

    def __init__(self, script, model=None):
        super().__init__(script, model)
        self.instructions_seen: list[str] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        seen = [str(options.get("instructions") or "")]
        seen += [m.text for m in messages if m.role == "system"]
        self.instructions_seen.append("\n".join(seen))
        return super()._inner_get_response(messages=messages, stream=stream, options=options,
                                           **kwargs)


def test_spawned_subagent_told_its_grants(tmp_path):
    a = InstrRec([text("a-done")])
    script = [spawn("a", 1, write_to=["out"]), wait([1], 1), results(1, 1), text("done")]
    run_tree(tmp_path, script, Queue(a))
    assert any("agents/a/" in t and "out" in t for t in a.instructions_seen)


def test_refusal_reaches_the_model(tmp_path):
    # The tool's own message, not the framework's bare "Function failed.".
    got: dict = {}
    script = [tool_call("write_file", {"path": ".sheath/x", "content": "c"}, "w1"),
              tool_call("read_file", {"path": "missing-dir/../../escape"}, "r1"), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
            agent = await s.create_agent(Rec(script), bundles=("files",))
            await agent.run("go", session=agent.create_session())

    asyncio.run(asyncio.wait_for(main(), 5))
    assert got["w1"].startswith("Error: ") and "managed by sheath" in got["w1"]
    assert got["r1"].startswith("Error: ") and "escapes root" in got["r1"]


def test_run_python_refusal_reaches_the_model(tmp_path):
    got: dict = {}
    script = [tool_call("run_python", {}, "p1"), record(got)]

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as s:
            agent = await s.create_agent(Rec(script), bundles=("code",))
            await agent.run("go", session=agent.create_session())

    asyncio.run(asyncio.wait_for(main(), 5))
    assert got["p1"] == "Error: run_python requires either code= or path="
