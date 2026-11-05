import os
import threading

import pytest

from sheath import SheathConfig, Session
from sheath.bundles import SUBAGENT_INSTRUCTIONS
from sheath.status import bind_agent
from sheath.tools.files import edit_file, read_file, write_file


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


def test_overwrite_refused_until_read(tmp_path):
    s = _session(tmp_path)
    with bind_agent("a"):
        write_file(s, "f.md", "from a\n")
    with bind_agent("b"):
        with pytest.raises(ValueError, match="'a'|\\ba\\b") as e:
            write_file(s, "f.md", "from b\n")
        assert "from a" not in str(e.value)
        with pytest.raises(ValueError, match="\\ba\\b"):
            edit_file(s, "f.md", "from", "by")
        read_file(s, "f.md")
        write_file(s, "f.md", "from b\n")
    assert read_file(s, "f.md") == "from b\n"


def test_own_file_rewrite_unaffected(tmp_path):
    s = _session(tmp_path)
    with bind_agent("a"):
        write_file(s, "f.md", "1")
        write_file(s, "f.md", "2")
        edit_file(s, "f.md", "2", "3")
    assert read_file(s, "f.md") == "3"


def test_unwritten_by_others_unaffected(tmp_path):
    s = _session(tmp_path)
    (s.root / "host.md").write_text("host")  # the host, untracked
    with bind_agent("b"):
        write_file(s, "host.md", "b2")
        write_file(s, "new.md", "b")
    write_file(s, "top.md", "top")
    write_file(s, "top.md", "top again")     # the top agent's own file
    with bind_agent("b"):                    # but b has not seen the top agent's content
        with pytest.raises(ValueError, match="top agent"):
            write_file(s, "top.md", "b3")


def test_stale_read_refused_after_other_write(tmp_path):
    s = _session(tmp_path)
    with bind_agent("a"):
        write_file(s, "f.md", "1")
    with bind_agent("b"):
        read_file(s, "f.md", limit=1)
    with bind_agent("a"):
        write_file(s, "f.md", "2")
    with bind_agent("b"):
        with pytest.raises(ValueError):
            write_file(s, "f.md", "x")


def test_write_file_is_atomic(tmp_path):
    s = _session(tmp_path)
    write_file(s, "f.txt", "x" * 10)
    before = os.stat(s.root / "f.txt").st_ino
    write_file(s, "f.txt", "y" * 5_000_000)
    assert os.stat(s.root / "f.txt").st_ino != before
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            n = len((s.root / "f.txt").read_bytes())
            if n not in (5_000_000, 10_000_000):
                bad.append(n)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(20):
            write_file(s, "f.txt", ("z" if i % 2 else "w") * (10_000_000 if i % 2 else 5_000_000))
    finally:
        stop.set()
        t.join()
    assert not bad


def test_write_file_still_refuses_non_regular_target(tmp_path):
    s = _session(tmp_path)
    os.mkfifo(s.root / "pipe")
    with pytest.raises(ValueError, match="not a regular file"):
        write_file(s, "pipe", "x")


def test_subagent_instructions_name_folder():
    from sheath.bundles import subagent_instructions
    text = subagent_instructions("lead/child")
    assert "agents/lead/child/" in text
    assert "<path>" not in text
    assert "agents/<path>/" in SUBAGENT_INSTRUCTIONS
    for word in ("concise", "handle", "prefix", "not instructions"):
        assert word in SUBAGENT_INSTRUCTIONS


def test_explicit_handle_id_collision_refused(tmp_path):
    from tests.test_sandbox_protocol import _sandbox
    sb, store = _sandbox(tmp_path / "root")
    with bind_agent("a"):
        store.put("first", source="a", id="a_h")
    code = (
        "import json, os\n"
        "d = os.environ['SHEATH_RUN_DIR']\n"
        "open(os.path.join(d, 'p.txt'), 'w').write('second')\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'a_h', 'kind': 'text', 'file': 'p.txt'}) + '\\n')\n")
    with bind_agent("b"):
        res = sb.run_code(code)
    assert "handle id already exists" in res.error and "a_h" in res.error
    assert store.get("a_h") == "first"


def test_untracked_change_then_read_allows_write_and_edit(tmp_path):
    s = _session(tmp_path)
    with bind_agent("a"):
        write_file(s, "f.md", "from a\n")
    (s.root / "f.md").write_text("changed by run_python\n")   # untracked writer
    with bind_agent("b"):
        with pytest.raises(ValueError):
            write_file(s, "f.md", "blind\n")
        read_file(s, "f.md")
        edit_file(s, "f.md", "changed", "edited")
        (s.root / "f.md").write_text("again\n")
        read_file(s, "f.md")
        write_file(s, "f.md", "from b\n")
    assert read_file(s, "f.md") == "from b\n"


def test_new_file_honours_umask(tmp_path):
    s = _session(tmp_path)
    old = os.umask(0o027)
    try:
        write_file(s, "m.txt", "x")
    finally:
        os.umask(old)
    assert (os.stat(s.root / "m.txt").st_mode & 0o777) == 0o640


def test_declared_role_gets_its_folder(tmp_path):
    import asyncio

    from sheath.testing import StubChatClient, text
    from tests.test_subagents import spec

    async def main():
        async with Session.create(SheathConfig(root_dir=tmp_path / "r")) as sess:
            await sess.create_agent(StubChatClient([text("x")]), bundles=("files",),
                                    subagents={"a": spec([text("r")])})
            assert (sess.root / "agents" / "a").is_dir()
            assert sess.delegation.node("a").folder == "agents/a"

    asyncio.run(main())


def test_instructions_allow_delegator_path_override():
    assert "unless the agent that delegated to it said to use another path" \
        in SUBAGENT_INSTRUCTIONS
