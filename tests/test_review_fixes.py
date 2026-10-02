"""Final-review fixes: no sandboxed code runs concurrently with parent-side file work, timed-out
containers are killed, publication hands the host a private snapshot, and the publish control
file cannot be hijacked or used to hang the parent."""

import re
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from tether import Session, TetherConfig
from tether.config import SandboxConfig
from tether.handles import HandleStore
from tether.publish import PublishError, validate_publication
from tether.sandbox_container import ContainerSandbox


def _session(tmp_path, host=None):
    return Session.create(TetherConfig(root_dir=tmp_path / "r"), on_publish=host)


def _tool(sess, name):
    return next(t for t in sess.tools("code", "deliver") if t.__name__ == name)


def _while_running(sess, code, action):
    """Start run_code(code) in a thread, run ``action`` once it is underway, and return
    (run_finished_at, action_finished_at)."""
    marks = {}

    def run():
        sess.sandbox.run_code(code)
        marks["run"] = time.monotonic()

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)                       # the child is now sleeping inside the sandbox
    action()
    marks["action"] = time.monotonic()
    t.join(30)
    return marks["run"], marks["action"]


_SLOW = "import time\ntime.sleep(1.0)\n"


# --- fix 2: parent-side file work is serialized with sandbox runs --------------------------

def test_publish_waits_for_a_running_sandbox(tmp_path):
    sess = _session(tmp_path, lambda pf: {})
    (sess.root / "outputs").mkdir()
    (sess.root / "outputs/a.csv").write_text("x")
    run_done, action_done = _while_running(
        sess, _SLOW, lambda: _tool(sess, "publish_file")("outputs/a.csv"))
    assert action_done >= run_done


def test_publish_handle_waits_for_a_running_sandbox(tmp_path):
    sess = _session(tmp_path, lambda pf: {})
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    run_done, action_done = _while_running(
        sess, _SLOW, lambda: _tool(sess, "publish_handle")(h.id))
    assert action_done >= run_done


def test_add_input_waits_for_a_running_sandbox(tmp_path):
    sess = _session(tmp_path)
    run_done, action_done = _while_running(
        sess, _SLOW, lambda: sess.add_input(b"x", input_id="f1", name="a.txt"))
    assert action_done >= run_done


def test_sandbox_runs_do_not_overlap(tmp_path):
    sess = _session(tmp_path)
    run_done, action_done = _while_running(sess, _SLOW, lambda: sess.sandbox.run_code("1"))
    assert action_done >= run_done


# --- fix 1: a timed-out container is removed, not orphaned ---------------------------------

def test_timed_out_container_is_removed(tmp_path, monkeypatch):
    import tether.container_runtime as cr
    import tether.sandbox_container as sc

    monkeypatch.setattr(cr, "ensure_image", lambda *a, **k: None)
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    sb = ContainerSandbox(root=tmp_path, store=HandleStore(tmp_path), runtime="podman",
                          config=SandboxConfig(backend="container", timeout_s=1))
    res = sb.run_code("1")
    assert res.killed_by == "timeout"
    run_argv = next(a for a in calls if a[1] == "run")
    name = run_argv[run_argv.index("--name") + 1]
    assert name.startswith("tether-")
    assert ["podman", "rm", "-f", name] in calls


# --- fix 3: the host gets a private snapshot -----------------------------------------------

def test_callback_receives_private_snapshot_removed_afterwards(tmp_path):
    seen = {}

    def host(pf):
        seen["path"] = pf.path
        seen["data"] = pf.path.read_bytes()
        seen["inside_root"] = pf.path.resolve().is_relative_to(sess.root)
        (sess.root / "outputs/a.csv").write_text("tampered")   # the original may change now
        seen["after_tamper"] = pf.path.read_bytes()
        return {}

    sess = _session(tmp_path, host)
    (sess.root / "outputs").mkdir()
    (sess.root / "outputs/a.csv").write_text("original")
    out = _tool(sess, "publish_file")("outputs/a.csv")
    assert "error" not in out
    assert seen["data"] == seen["after_tamper"] == b"original"
    assert seen["inside_root"] is False
    assert not seen["path"].exists()


# --- fix 4: the publish control file ------------------------------------------------------

def test_control_file_name_is_unpredictable(tmp_path):
    # The control files live in a per-run exchange directory named by a random token.
    sess = _session(tmp_path, lambda pf: {})
    res = sess.sandbox.run_code("import os\nos.environ['TETHER_RUN_DIR']")
    run_dir = Path(res.result)
    assert re.fullmatch(r"[0-9a-f]{16}", run_dir.name), res.result
    assert run_dir.parent == sess.root / ".tether" / "runs"


def test_fifo_control_file_does_not_hang_parent(tmp_path):
    sess = _session(tmp_path, lambda pf: {})
    code = ("import os\np = os.path.join(os.environ['TETHER_RUN_DIR'], 'publish.jsonl')\nos.mkfifo(p)\n")
    out = {}
    t = threading.Thread(target=lambda: out.update(res=sess.sandbox.run_code(code)), daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "parent hung reading a FIFO control file"
    assert "publish control file not a regular file" in (out["res"].error or "")


def test_symlinked_control_file_is_not_followed(tmp_path):
    calls = []
    sess = _session(tmp_path, lambda pf: calls.append(pf) or {})
    (sess.root / "outputs").mkdir()
    (sess.root / "outputs/a.csv").write_text("x")
    forged = tmp_path / "forged.jsonl"
    forged.write_text('{"path": "outputs/a.csv"}\n')
    code = (f"import os\np = os.path.join(os.environ['TETHER_RUN_DIR'], 'publish.jsonl')\n"
            f"os.symlink({str(forged)!r}, p)\n")
    res = sess.sandbox.run_code(code)
    assert calls == []
    assert "publish control file not a regular file" in (res.error or "")


# --- publication path errors never escape as exceptions ------------------------------------

@pytest.mark.parametrize("bad", ["a\x00b", "\ud800.csv"])
def test_unencodable_paths_raise_publish_error(tmp_path, bad):
    with pytest.raises(PublishError):
        validate_publication(tmp_path, bad, name=None, description=None, source="s",
                             max_bytes=1024)


def test_nul_path_from_sandbox_is_a_record_not_a_crash(tmp_path):
    sess = _session(tmp_path, lambda pf: {})
    code = ("import json, os\nwith open(os.path.join(os.environ['TETHER_RUN_DIR'], 'publish.jsonl'), 'a') as f:\n"
            "    f.write(json.dumps({'path': 'a\\u0000b'}) + '\\n')\n"
            "from tether_sandbox import save\nsave('h7', {'k': 1})\n")
    res = sess.sandbox.run_code(code)
    assert res.published and "error" in res.published[0]
    assert "h7" in res.new_handles
