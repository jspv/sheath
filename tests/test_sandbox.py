import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tether.config import SandboxConfig
from tether.handles import HandleStore
from tether.paths import PathEscapesRootError
from tether.sandbox import ControlPlaneLimits, ExecResult, LocalSubprocessSandbox

RUNTIME_DIR =Path(__file__).resolve().parent.parent / "tether" / "runtime"


def _run_child(tmp_path, body: str, registry: dict | None = None, args: list[str] | None = None):
    """Run a small script in a child process with the helper env wired up."""
    root = tmp_path
    (root / "handles").mkdir(exist_ok=True)
    run_dir = root / ".tether" / "runs" / "t"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "registry.json").write_text(json.dumps(registry or {}))
    new_handles = run_dir / "new_handles.jsonl"
    emit = run_dir / "emit.json"

    script = root / "script.py"
    script.write_text(body)

    env = {
        "PATH": os.environ.get("PATH", ""),
        "TETHER_ROOT": str(root),
        "TETHER_RUN_DIR": str(run_dir),
        "PYTHONPATH": str(RUNTIME_DIR),
    }
    proc = subprocess.run(
        [sys.executable, str(script), *(args or [])],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    return proc, new_handles, emit


def test_helper_emit_writes_payload(tmp_path):
    proc, _, emit = _run_child(
        tmp_path,
        "from tether_sandbox import emit\nemit({'total': 42})\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(emit.read_text()) == {"total": 42}


def test_helper_save_records_new_handle(tmp_path):
    proc, new_handles, _ = _run_child(
        tmp_path,
        "from tether_sandbox import save\nsave('h5', 'derived text')\n",
    )
    assert proc.returncode == 0, proc.stderr
    line = json.loads(new_handles.read_text().strip())
    assert line["id"] == "h5"
    assert line["kind"] == "text"
    assert "path" not in line                      # the child names a file, not a path
    assert (new_handles.parent / line["file"]).read_text() == "derived text"


def test_helper_load_reads_existing_text_handle(tmp_path):
    (tmp_path / "handles").mkdir(exist_ok=True)
    (tmp_path / "handles" / "h1.txt").write_text("input data")
    registry = {"h1": {"kind": "text", "path": "handles/h1.txt"}}
    proc, _, emit = _run_child(
        tmp_path,
        "from tether_sandbox import load, emit\nemit(load('h1'))\n",
        registry=registry,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(emit.read_text()) == "input data"


def test_helper_passes_argv(tmp_path):
    proc, _, emit = _run_child(
        tmp_path,
        "import sys\nfrom tether_sandbox import emit\nemit(sys.argv[1:])\n",
        args=["EU", "2025"],
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(emit.read_text()) == ["EU", "2025"]


def _sandbox(tmp_path):
    store = HandleStore(tmp_path)
    return LocalSubprocessSandbox(root=tmp_path, store=store, config=SandboxConfig()), store


def test_run_script_captures_emit_result(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text(
        "from tether_sandbox import emit\nemit({'ok': True})\n"
    )
    res = sb.run_script("s.py")
    assert isinstance(res, ExecResult)
    assert res.exit_code == 0
    assert res.result == {"ok": True}
    assert res.error is None


def test_run_script_captures_stdout(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text("print('hello from child')\n")
    res = sb.run_script("s.py")
    assert "hello from child" in res.stdout


def test_run_script_reports_new_handles_and_registers_them(tmp_path):
    sb, store = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text(
        "from tether_sandbox import save\nsave('h1', {'derived': 1})\n"
    )
    res = sb.run_script("s.py")
    assert res.new_handles == ["h1"]
    assert store.get("h1") == {"derived": 1}  # parent ingested it


def test_run_script_can_load_existing_handle(tmp_path):
    sb, store = _sandbox(tmp_path)
    store.put({"input": 99}, source="seed", id="h1")
    (tmp_path / "s.py").write_text(
        "from tether_sandbox import load, emit\nemit(load('h1'))\n"
    )
    res = sb.run_script("s.py")
    assert res.result == {"input": 99}


def test_run_script_passes_args(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text(
        "import sys\nfrom tether_sandbox import emit\nemit(sys.argv[1:])\n"
    )
    res = sb.run_script("s.py", args=["EU", "2025"])
    assert res.result == ["EU", "2025"]


def test_run_script_captures_exception_as_error(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text("raise ValueError('boom')\n")
    res = sb.run_script("s.py")
    assert res.exit_code != 0
    assert "ValueError: boom" in res.error


def test_run_script_times_out(tmp_path):
    store = HandleStore(tmp_path)
    sb = LocalSubprocessSandbox(root=tmp_path, store=store,
                                config=SandboxConfig(timeout_s=0.5))
    (tmp_path / "s.py").write_text("import time\ntime.sleep(5)\n")
    res = sb.run_script("s.py")
    assert res.killed_by == "timeout"
    assert res.exit_code != 0


def test_run_script_rejects_path_outside_root(tmp_path):
    sb, _ = _sandbox(tmp_path)
    with pytest.raises(PathEscapesRootError):
        sb.run_script("../evil.py")


def test_run_code_convenience_writes_and_runs_inline(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code("from tether_sandbox import emit\nemit(7)\n")
    assert res.result == 7


# --- robustness: a misbehaving child must yield an ExecResult, never raise ---

def test_malformed_emit_payload_is_graceful(tmp_path):
    sb, _ = _sandbox(tmp_path)
    # Child exits 0 but writes garbage directly to the emit file.
    (tmp_path / "s.py").write_text(
        "import os\nopen(os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'), 'w').write('{not json')\n"
    )
    res = sb.run_script("s.py")  # must not raise
    assert res.exit_code == 0
    assert res.result is None
    assert "malformed emit" in res.error


def test_corrupt_handle_line_is_skipped_not_fatal(tmp_path):
    sb, store = _sandbox(tmp_path)
    # One good handle, then a corrupt jsonl line appended directly.
    (tmp_path / "s.py").write_text(
        "import os\n"
        "from tether_sandbox import save\n"
        "save('h1', {'ok': 1})\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a').write('{bad json\\n')\n"
    )
    res = sb.run_script("s.py")  # must not raise
    assert res.new_handles == ["h1"]      # good one reported
    assert store.get("h1") == {"ok": 1}   # and registered
    assert res.exit_code == 0


def test_dataframe_roundtrip_through_sandbox(tmp_path):
    import pandas as pd

    sb, store = _sandbox(tmp_path)
    store.put(pd.DataFrame({"revenue": [120, 0, 210, 0, 95]}), source="seed", id="h1")
    (tmp_path / "s.py").write_text(
        "from tether_sandbox import load, save, emit\n"
        "df = load('h1')\n"
        "clean = df[df.revenue > 0]\n"
        "save('h2', clean)\n"
        "emit({'total': int(clean.revenue.sum()), 'dropped': int((df.revenue <= 0).sum())})\n"
    )
    res = sb.run_script("s.py")
    assert res.result == {"total": 425, "dropped": 2}
    assert res.new_handles == ["h2"]
    assert list(store.get("h2").revenue) == [120, 210, 95]


def test_emits_nothing_yields_none_result_and_no_error(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text("x = 1 + 1\n")
    res = sb.run_script("s.py")
    assert res.exit_code == 0
    assert res.result is None
    assert res.error is None


def test_stderr_on_success_is_captured_without_setting_error(tmp_path):
    sb, _ = _sandbox(tmp_path)
    (tmp_path / "s.py").write_text("import sys\nsys.stderr.write('just a warning')\n")
    res = sb.run_script("s.py")
    assert res.exit_code == 0
    assert "just a warning" in res.stderr
    assert res.error is None


def test_inline_scripts_do_not_collide_across_instances(tmp_path):
    store = HandleStore(tmp_path)
    sb_a = LocalSubprocessSandbox(root=tmp_path, store=store, config=SandboxConfig())
    sb_b = LocalSubprocessSandbox(root=tmp_path, store=store, config=SandboxConfig())
    res_a = sb_a.run_code("from tether_sandbox import emit\nemit('A')\n")
    res_b = sb_b.run_code("from tether_sandbox import emit\nemit('B')\n")
    assert (res_a.result, res_b.result) == ("A", "B")


def test_last_expression_is_auto_emitted(tmp_path):
    # The model can write a bare expression (Jupyter style) and get a result back,
    # without knowing about emit().
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code("x = 40\nx + 2\n")
    assert res.result == 42


def test_print_falls_back_to_stdout_when_no_emit(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code("print('the answer is 144')\n")
    assert res.result == "the answer is 144"


def test_explicit_emit_takes_precedence_over_last_expr(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code("from tether_sandbox import emit\nemit(1)\n2 + 2\n")
    assert res.result == 1


def test_last_expression_can_load_a_handle(tmp_path):
    sb, store = _sandbox(tmp_path)
    store.put({"v": [1, 2, 3]}, source="seed", id="h1")
    res = sb.run_code("from tether_sandbox import load\nsum(load('h1')['v'])\n")
    assert res.result == 6


def test_child_record_carries_no_metadata(tmp_path):
    # Assert what the CHILD wrote, not what the parent passed on: the parent hard-codes
    # the four kwargs, so spying on adopt() proves nothing about the child.
    sb, store = _sandbox(tmp_path)
    captured = []
    original_ingest = type(sb)._ingest_new_handles

    def spy(self, run_dir):
        for line in (run_dir / "new_handles.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                captured.append(json.loads(line))
        return original_ingest(self, run_dir)

    type(sb)._ingest_new_handles = spy
    try:
        res = sb.run_code(
            "import pandas as pd\n"
            "from tether_sandbox import save\n"
            "save('h1', pd.DataFrame({'a': [1, 2, 3]}))\n")
    finally:
        type(sb)._ingest_new_handles = original_ingest

    assert res.error is None, res.error
    assert [set(rec) for rec in captured] == [{"id", "kind", "file", "source"}]
    assert store.summary("h1")["n_rows"] == 3      # parent still derived the truth


def test_forged_child_metadata_is_overridden(tmp_path):
    """A child writing a record by hand cannot make the store report false metadata."""
    sb, store = _sandbox(tmp_path)
    res = sb.run_code(
        "import json, os\n"
        "import pandas as pd\n"
        "pd.DataFrame({'a': range(1000)}).to_parquet(os.path.join(os.environ['TETHER_RUN_DIR'], 'h1.parquet'))\n"
        "rec = {'id': 'h1', 'kind': 'dataframe', 'file': 'h1.parquet',\n"
        "       'source': 'run_python', 'n_rows': 2, 'preview': 'all clean!',\n"
        "       'bytes': 10, 'schema': {'a': 'string'}}\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a').write(json.dumps(rec) + '\\n')\n"
    )
    assert res.error is None, res.error
    summary = store.summary("h1")
    assert summary["n_rows"] == 1000              # not the claimed 2
    assert summary["preview"] != "all clean!"
    assert summary["schema"] == {"a": "int64"}    # not the claimed string
    assert summary["bytes"] > 10


def test_child_cannot_repoint_an_existing_handle(tmp_path):
    """Adopting an existing id is refused (and reported); ingestion stays tolerant."""
    sb, store = _sandbox(tmp_path)
    original = store.put({"trusted": True}, source="parent")
    res = sb.run_code(
        "import json, os\n"
        f"rec = {{'id': {original.id!r}, 'kind': 'text', 'file': 'evil.txt',\n"
        "       'source': 'run_python'}\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'evil.txt'), 'w').write('attacker data')\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a').write(json.dumps(rec) + '\\n')\n"
    )
    assert "already exists" in (res.error or "")                   # refused and surfaced
    assert res.new_handles == []                                  # rejected
    assert store.summary(original.id) == original.summary()       # untouched


def test_one_corrupt_record_does_not_abort_ingestion(tmp_path):
    sb, store = _sandbox(tmp_path)
    res = sb.run_code(
        "import json, os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'a.txt'), 'w').write('a')\n"
        "f = open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a')\n"
        "f.write('{not json\\n')\n"
        "f.write(json.dumps({'id': 'h1', 'kind': 'text', 'file': 'a.txt',\n"
        "                    'source': 'run_python'}) + '\\n')\n"
        "f.close()\n"
    )
    assert res.exit_code == 0
    assert res.new_handles == ["h1"]
    # The corrupt line is reported, not silently dropped -- and only it.
    assert res.error == ("tether: 1 handle record(s) rejected (invalid id or kind, or the "
                         "saved file was missing or unreadable)")


def _tiny_limits_sandbox(tmp_path, **kw):
    root = tmp_path / "r"
    store = HandleStore(root)
    limits = ControlPlaneLimits(**{"max_emit_bytes": 1024, "max_control_bytes": 1024,
                                   "max_new_handles": 2, **kw})
    return LocalSubprocessSandbox(root=root, store=store, config=SandboxConfig(),
                                  limits=limits), store


def test_oversized_emit_is_rejected_not_parsed(tmp_path):
    sb, _ = _tiny_limits_sandbox(tmp_path)
    # Invalid JSON: only passes if the size check runs before parsing.
    res = sb.run_code(
        "import os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'), 'w').write('{' * 50_000)\n")
    assert res.result is None
    assert "emit payload too large" in (res.error or "")
    assert "malformed" not in res.error


def test_new_handles_record_count_is_capped(tmp_path):
    sb, _ = _tiny_limits_sandbox(tmp_path, max_control_bytes=1024 * 1024)
    res = sb.run_code(
        "from tether_sandbox import save\n"
        "for i in range(10):\n"
        "    save(f'h{i}', {'i': i})\n"
    )
    assert len(res.new_handles) == 2                 # max_new_handles
    assert "too many new handles" in (res.error or "")


def test_oversized_new_handles_file_is_bounded(tmp_path):
    sb, _ = _tiny_limits_sandbox(tmp_path, max_new_handles=1000)
    res = sb.run_code(
        "import os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a').write('x' * 20_000)\n"
    )
    assert "control file too large" in (res.error or "")


def test_resaving_an_existing_id_raises_in_the_child(tmp_path):
    # Honest code must fail loudly at the point of the mistake, not believe it succeeded.
    sb, store = _sandbox(tmp_path)
    sb.run_code("from tether_sandbox import save\nsave('h1', {'v': 1})\n")
    res = sb.run_code("from tether_sandbox import save\nsave('h1', {'v': 2})\n")
    assert res.exit_code != 0
    assert "already exists" in (res.error or "")


def test_id_reuse_rejection_is_reported_not_silent(tmp_path):
    # The parent is the boundary: a hand-written control record reusing an id is refused
    # AND surfaced, so stale metadata can never sit silently over changed bytes.
    sb, store = _sandbox(tmp_path)
    original = store.put({"trusted": True}, source="parent")
    res = sb.run_code(
        "import json, os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'evil.txt'), 'w').write('attacker data')\n"
        f"rec = {{'id': {original.id!r}, 'kind': 'text', 'file': 'evil.txt',\n"
        "       'source': 'run_python'}\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a').write(json.dumps(rec) + '\\n')\n")
    assert res.new_handles == []
    assert "already exists" in (res.error or "")
    assert store.summary(original.id) == original.summary()


def test_resaving_an_id_within_one_run_also_raises(tmp_path):
    # The registry is an import-time snapshot, so the child must also track what
    # this run has already written.
    sb, store = _sandbox(tmp_path)
    res = sb.run_code(
        "from tether_sandbox import save\n"
        "save('h1', {'v': 1})\n"
        "save('h1', {'v': 2})\n"
    )
    assert res.exit_code != 0
    assert "already exists" in (res.error or "")


def test_adopt_raises_a_typed_error_on_id_reuse(tmp_path):
    # Ingestion dispatches on this type, not on message text.
    from tether.handles import HandleIdReuseError

    store = HandleStore(tmp_path)
    h = store.put({"a": 1}, source="parent")
    (tmp_path / "handles" / "x.txt").write_text("x")
    with pytest.raises(HandleIdReuseError):
        store.adopt(id=h.id, kind="text", path="handles/x.txt", source="s")
    assert issubclass(HandleIdReuseError, ValueError)


def test_id_reuse_reporting_is_bounded(tmp_path):
    # Surfacing rejections must not itself become a flood channel: the error string
    # cannot grow with the number of rejected records.
    sb, store = _sandbox(tmp_path)
    original = store.put({"trusted": True}, source="parent")
    res = sb.run_code(
        "import json, os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'e.txt'), 'w').write('x')\n"
        f"rec = json.dumps({{'id': {original.id!r}, 'kind': 'text',\n"
        "                   'file': 'e.txt', 'source': 'run_python'})\n"
        "f = open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a')\n"
        "for _ in range(5000):\n"
        "    f.write(rec + '\\n')\n"
        "f.close()\n")
    assert res.new_handles == []
    assert "5000" in (res.error or "")        # the count is reported
    assert len(res.error) < 1000              # but the message stays small


def test_id_reuse_reporting_bounds_distinct_and_long_ids(tmp_path):
    sb, store = _sandbox(tmp_path)
    for i in range(10):
        store.put({"i": i}, source="parent", id=f"{'x' * 100}{i}")
    res = sb.run_code(
        "import json, os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'e.txt'), 'w').write('x')\n"
        "f = open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'a')\n"
        "for i in range(10):\n"
        "    f.write(json.dumps({'id': 'x' * 100 + str(i), 'kind': 'text',\n"
        "                        'file': 'e.txt', 'source': 's'}) + '\\n')\n"
        "f.close()\n")
    assert "10" in res.error
    assert len(res.error) < 1000


def test_session_create_carries_config_caps_into_sandbox(tmp_path):
    from tether.config import TetherConfig
    from tether.session import Session

    cfg = TetherConfig(root_dir=tmp_path / "s", max_emit_bytes=11, max_control_bytes=22,
                       max_new_handles=3)
    lim = Session.create(cfg).sandbox.limits
    assert (lim.max_emit_bytes, lim.max_control_bytes, lim.max_new_handles) == (11, 22, 3)


def test_container_sandbox_stores_limits(tmp_path):
    from tether.sandbox_container import ContainerSandbox

    limits = ControlPlaneLimits(max_emit_bytes=5, max_control_bytes=6, max_new_handles=7)
    sb = ContainerSandbox(root=tmp_path, store=None, runtime="podman", limits=limits)
    assert sb.limits is limits


# --- the control channel must never raise out of run_python -------------------------------
# run_python is documented to return {stdout, stderr, result, error, exit_code, new_handles,
# killed_by}. Anything a hostile or merely broken child can do to a control file has to come
# back as that dict with an `error`, not as an exception unwinding through the tool layer.

def _assert_is_exec_result(res):
    for field in ("stdout", "stderr", "result", "error", "exit_code", "new_handles",
                  "killed_by"):
        assert hasattr(res, field), f"missing {field}"


def test_non_utf8_emit_payload_is_reported_not_raised(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code(
        "import os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'), 'wb').write(b'\\xff\\xfe\\x00bad')\n"
    )
    _assert_is_exec_result(res)
    assert res.result is None
    assert "malformed emit payload" in (res.error or "")


def test_non_utf8_new_handles_record_is_reported_not_raised(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code(
        "import os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'wb').write(b'\\xff\\xfe\\x00bad\\n')\n"
    )
    _assert_is_exec_result(res)
    assert res.new_handles == []


def test_non_utf8_bytes_do_not_hide_a_valid_new_handle_record(tmp_path):
    """errors='replace' must degrade the bad line only -- a good record still lands."""
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code(
        "import json, os\n"
        "open(os.path.join(os.environ['TETHER_RUN_DIR'], 'ok.txt'), 'w').write('fine')\n"
        "f = open(os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl'), 'ab')\n"
        "f.write(b'\\xff\\xfe garbage\\n')\n"
        "f.write((json.dumps({'id': 'h9', 'kind': 'text', 'file': 'ok.txt',\n"
        "                     'source': 'run_python'}) + '\\n').encode())\n"
        "f.close()\n"
    )
    _assert_is_exec_result(res)
    assert res.new_handles == ["h9"]


def test_directory_at_the_emit_path_is_reported_not_raised(tmp_path):
    """A child that pre-creates a directory at a control path makes read_text raise
    IsADirectoryError, and would then make the finally-block unlink raise while unwinding."""
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code(
        "import os\n"
        "os.mkdir(os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'))\n"
    )
    _assert_is_exec_result(res)
    assert res.result is None
    assert "emit payload not a regular file" in (res.error or "")


def test_directory_at_the_new_handles_path_is_reported_not_raised(tmp_path):
    sb, _ = _sandbox(tmp_path)
    res = sb.run_code(
        "import os\n"
        "p = os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl')\n"
        "os.mkdir(p)\n"
    )
    _assert_is_exec_result(res)
    assert res.exit_code == 0, res.error
    assert res.new_handles == []
    assert "control file not a regular file" in (res.error or "")


def test_control_plane_limit_defaults_match_tether_config(tmp_path):
    """ControlPlaneLimits duplicates TetherConfig's caps with only a docstring holding them
    together. Nothing but this test notices if one side is changed alone."""
    from tether.config import TetherConfig

    cfg = TetherConfig()
    lim = ControlPlaneLimits()
    assert (lim.max_emit_bytes, lim.max_control_bytes, lim.max_new_handles) == (
        cfg.max_emit_bytes, cfg.max_control_bytes, cfg.max_new_handles)
