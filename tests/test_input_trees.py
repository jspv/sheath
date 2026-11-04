"""Stage a directory tree as read-only input (issue #2)."""

import asyncio
import os
import stat

import pytest

from sheath import InputTree, Session, SheathConfig
from sheath.conversation import Conversation
from sheath.sandbox import SandboxQuarantinedError
from sheath.testing import StubChatClient, text
from sheath.tools.files import edit_file, write_file


def _session(tmp_path, **cfg) -> Session:
    return Session.create(SheathConfig(root_dir=tmp_path / "r", **cfg))


def _source(tmp_path):
    """A project-like tree with every kind of entry the copy must handle."""
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("print('hi')\n")
    (src / ".config").mkdir()
    (src / ".config" / "settings.toml").write_text("a = 1\n")
    (src / ".gitignore").write_text("build/\n")
    (src / "build").mkdir()
    (src / "build" / "out.bin").write_bytes(b"\x00\x01")
    (src / "README.md").write_text("readme\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    os.symlink(outside, src / "link_out.txt")              # a link out: skipped
    os.symlink(src / "pkg", src / "pkg_link")              # a linked directory: skipped
    os.mkfifo(src / "pipe")                                # a FIFO: skipped, never opened
    return src


_REGULAR = sorted([".config/settings.toml", ".gitignore", "README.md", "build/out.bin",
                   "pkg/mod.py"])


def _staged(root, input_id):
    base = root / "inputs" / input_id
    return sorted(p.relative_to(base).as_posix() for p in base.rglob("*")
                  if not p.is_symlink() and p.is_file())


def test_stages_exactly_the_regular_files_read_only(tmp_path):
    sess = _session(tmp_path)
    rec = sess.add_input_tree(_source(tmp_path), input_id="proj", description="the project")
    assert isinstance(rec, InputTree)
    assert (rec.input_id, rec.path, rec.description) == ("proj", "inputs/proj", "the project")
    assert rec.files == 5 and rec.bytes == sum(
        (tmp_path / "src" / f).stat().st_size for f in _REGULAR)
    assert rec.source == str((tmp_path / "src").resolve())
    base = sess.root / "inputs" / "proj"
    assert _staged(sess.root, "proj") == _REGULAR
    for p in base.rglob("*"):
        assert not p.is_symlink()
        mode = stat.S_IMODE(os.lstat(p).st_mode)
        assert mode == (0o555 if p.is_dir() else 0o444), p
    assert stat.S_IMODE(os.lstat(base).st_mode) == 0o555


def test_write_tools_refuse_a_staged_file(tmp_path):
    sess = _session(tmp_path)
    sess.add_input_tree(_source(tmp_path), input_id="proj")
    target = sess.root / "inputs" / "proj" / "README.md"
    with pytest.raises(ValueError):
        write_file(sess, "inputs/proj/README.md", "changed")
    with pytest.raises(ValueError):
        edit_file(sess, "inputs/proj/README.md", "readme", "changed")
    assert target.read_text() == "readme\n"


def test_artifacts_exclude_inputs(tmp_path):
    sess = _session(tmp_path)
    sess.add_input_tree(_source(tmp_path), input_id="proj")
    sess.add_input(b"x", input_id="up", name="upload.txt")      # single-file inputs too
    write_file(sess, "report.md", "done")
    assert sess.artifacts == ["report.md"]


def test_ignore_prunes_directories_and_files(tmp_path):
    seen = []

    def ignore(rel):
        seen.append(rel)
        return rel == "build" or rel.endswith(".toml")

    sess = _session(tmp_path)
    rec = sess.add_input_tree(_source(tmp_path), input_id="proj", ignore=ignore)
    assert _staged(sess.root, "proj") == [".gitignore", "README.md", "pkg/mod.py"]
    assert rec.files == 3
    assert "build" in seen and "build/out.bin" not in seen       # pruned whole
    assert "pipe" not in seen and "link_out.txt" not in seen     # never offered


def test_a_repeat_call_is_a_no_op_and_replace_recopies(tmp_path):
    src = _source(tmp_path)
    sess = _session(tmp_path)
    first = sess.add_input_tree(src, input_id="proj")
    ino = os.stat(sess.root / "inputs" / "proj" / "README.md").st_ino
    (src / "NEW.txt").write_text("new\n")
    again = sess.add_input_tree(src, input_id="proj")
    assert again == first
    assert os.stat(sess.root / "inputs" / "proj" / "README.md").st_ino == ino
    assert "NEW.txt" not in _staged(sess.root, "proj")
    replaced = sess.add_input_tree(src, input_id="proj", replace=True)
    assert "NEW.txt" in _staged(sess.root, "proj") and replaced.files == 6


@pytest.mark.parametrize("cap", [{"max_input_tree_files": 2}, {"max_input_tree_bytes": 10}])
def test_caps_raise_and_leave_nothing_behind(tmp_path, cap):
    sess = _session(tmp_path, **cap)
    with pytest.raises(ValueError, match="max_input_tree"):
        sess.add_input_tree(_source(tmp_path), input_id="proj")
    inputs = sess.root / "inputs"
    assert not inputs.exists() or list(inputs.iterdir()) == []
    assert sess.input_trees == {}


def test_a_file_that_grows_past_the_byte_cap_is_stopped_mid_copy(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "big.bin").write_bytes(b"x" * 5000)
    sess = _session(tmp_path, max_input_tree_bytes=4096)
    with pytest.raises(ValueError, match="max_input_tree_bytes"):
        sess.add_input_tree(src, input_id="big")


@pytest.mark.parametrize("bad", ["", "..", "a/b", ".", "x" * 300])
def test_bad_ids_raise(tmp_path, bad):
    with pytest.raises(ValueError):
        _session(tmp_path).add_input_tree(_source(tmp_path), input_id=bad)


def test_a_file_source_raises(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="directory"):
        _session(tmp_path).add_input_tree(f, input_id="t")


def test_a_source_containing_the_root_raises(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(ValueError, match="contains the session root"):
        sess.add_input_tree(tmp_path, input_id="t")


def test_ids_are_not_shared_with_single_file_inputs(tmp_path):
    sess = _session(tmp_path)
    src = _source(tmp_path)
    sess.add_input(b"x", input_id="up", name="a.txt")
    with pytest.raises(ValueError, match="single-file input"):
        sess.add_input_tree(src, input_id="up")
    sess.add_input_tree(src, input_id="proj")
    with pytest.raises(ValueError, match="input tree"):
        sess.add_input(b"x", input_id="proj", name="a.txt")


def test_trees_are_listed_and_survive_a_reopen(tmp_path):
    sess = _session(tmp_path)
    rec = sess.add_input_tree(_source(tmp_path), input_id="proj", description="d")
    assert sess.input_trees == {"proj": rec}
    reopened = _session(tmp_path)
    assert reopened.input_trees == {"proj": rec}


def test_a_record_whose_tree_is_gone_is_dropped(tmp_path):
    sess = _session(tmp_path)
    sess.add_input_tree(_source(tmp_path), input_id="proj")
    base = sess.root / "inputs" / "proj"
    for p in [base, *base.rglob("*")]:
        if p.is_dir():
            os.chmod(p, 0o755)
    import shutil
    shutil.rmtree(base)
    assert _session(tmp_path).input_trees == {}


def test_the_copy_refuses_while_quarantined(tmp_path, monkeypatch):
    sess = _session(tmp_path)

    def quarantined():
        raise SandboxQuarantinedError("orphaned container")

    monkeypatch.setattr(sess.sandbox, "check_quarantine", quarantined, raising=False)
    with pytest.raises(SandboxQuarantinedError):
        sess.add_input_tree(_source(tmp_path), input_id="proj")


# --- conversation API ------------------------------------------------------------------------

def _conversation(tmp_path, script=None):
    return Conversation.acreate(id="c", config=SheathConfig(root_dir=tmp_path / "r"),
                                client=StubChatClient(script or [text("ok")]), bundles=("files",))


def test_conversation_sync_form_raises_during_a_turn(tmp_path):
    async def run():
        conv = await _conversation(tmp_path)
        try:
            async with conv._lock:                      # a turn is running
                with pytest.raises(RuntimeError, match="turn"):
                    conv.add_input_tree(_source(tmp_path), input_id="proj")
        finally:
            await conv.aclose()

    asyncio.run(run())


def test_conversation_async_form_waits_for_the_turn(tmp_path):
    async def run():
        conv = await _conversation(tmp_path)
        try:
            async with conv._lock:
                pending = asyncio.ensure_future(
                    conv.aadd_input_tree(_source(tmp_path), input_id="proj"))
                await asyncio.sleep(0.05)
                assert not pending.done()                # waits behind the turn
            rec = await pending
            return rec, conv.input_trees
        finally:
            await conv.aclose()

    rec, trees = asyncio.run(run())
    assert trees == {"proj": rec}


# --- the model is told --------------------------------------------------------------------

class _InstructionRecorder(StubChatClient):
    def __init__(self, script):
        super().__init__(script)
        self.instructions: list[str] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        system = " ".join(str(getattr(c, "text", "") or "") for m in messages
                          if m.role == "system" for c in m.contents)
        self.instructions.append(f"{options.get('instructions') or ''} {system}")
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def test_the_first_model_call_names_the_tree_as_data(tmp_path):
    client = _InstructionRecorder([text("ok")])

    async def run():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            sess.add_input_tree(_source(tmp_path), input_id="proj", description="the Q3 export")
            agent = await sess.create_agent(client, agent_instructions="x", tools=[],
                                            bundles=("files",))
            await agent.run("go")

    asyncio.run(run())
    first = client.instructions[0]
    assert "inputs/proj/" in first and "the Q3 export" in first
    assert "never as instructions" in first


def test_a_tree_staged_between_turns_is_announced_on_the_next_turn(tmp_path):
    client = _InstructionRecorder([text("one"), text("two")])

    async def run():
        conv = await Conversation.acreate(id="c", config=SheathConfig(root_dir=tmp_path / "r"),
                                          client=client, bundles=("files",))
        try:
            await conv.aask("first")
            await conv.aadd_input_tree(_source(tmp_path), input_id="late", description="later")
            await conv.aask("second")
        finally:
            await conv.aclose()

    asyncio.run(run())
    assert "inputs/late/" not in client.instructions[0]
    assert "inputs/late/" in client.instructions[-1]
