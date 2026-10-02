import json

import pytest

from tether.config import SandboxConfig
from tether.handles import HandleStore
from tether.sandbox import ControlPlaneLimits, LocalSubprocessSandbox
from tether.workspace import WorkspaceTamperedError


def _sandbox(root, **limits):
    store = HandleStore(root)
    sb = LocalSubprocessSandbox(root=root, store=store, config=SandboxConfig(backend="local"),
                                limits=ControlPlaneLimits(**limits) if limits else None)
    return sb, store


def _outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text(json.dumps({"key": "HOST-SECRET"}))
    victim = outside / "victim.txt"
    victim.write_text("original")
    return secret, victim


def test_emit_save_and_load_round_trip(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import pandas as pd\n"
        "from tether_sandbox import emit, save\n"
        "save('df1', pd.DataFrame({'a': [1, 2, 3]}))\n"
        "save('j1', {'k': 'v'})\n"
        "save('t1', 'hello')\n"
        "emit({'ok': True})\n")
    assert res.error is None, res.error
    assert res.result == {"ok": True}
    assert res.new_handles == ["df1", "j1", "t1"]
    assert store.get("j1") == {"k": "v"} and store.get("t1") == "hello"
    assert store.manifest_handles()["df1"].path == "handles/df1.parquet"
    res2 = sb.run_code("from tether_sandbox import load\nlen(load('df1'))\n")
    assert res2.result == 3


def test_run_directory_is_removed_after_the_run(tmp_path):
    root = tmp_path / "root"
    sb, _ = _sandbox(root)
    sb.run_code("from tether_sandbox import emit\nemit(1)\n")
    assert list((root / ".tether" / "runs").iterdir()) == []
    assert not any(p.name.startswith("_") for p in root.iterdir())


def test_emit_symlink_to_a_host_file_is_not_read(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import os\n"
        "p = os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json')\n"
        f"os.symlink({str(secret)!r}, p)\n")
    assert res.result is None
    assert "emit payload not a regular file" in res.error
    assert "HOST-SECRET" not in json.dumps([res.result, res.error, res.stdout])


def test_new_handles_symlink_is_not_read(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import os\n"
        "p = os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl')\n"
        f"os.symlink({str(secret)!r}, p)\n")
    assert res.new_handles == []
    assert "control file not a regular file" in res.error


def test_payload_symlink_is_rejected_and_reported(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        f"os.symlink({str(secret)!r}, os.path.join(d, 'x.json'))\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'leak', 'kind': 'json', 'file': 'x.json'}) + '\\n')\n")
    assert res.new_handles == []
    assert "1 handle record(s) rejected" in res.error
    assert "leak" not in store.manifest()
    assert secret.exists()                       # the link was not followed or moved


def test_payload_file_name_with_a_separator_is_rejected(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'x', 'kind': 'text', 'file': '../../manifest.json'}) + '\\n')\n")
    assert res.new_handles == [] and "rejected" in res.error
    assert store.manifest() == {}


def test_replacing_handles_dir_with_a_link_is_refused(tmp_path):
    _, victim = _outside(tmp_path)
    root = tmp_path / "root"
    sb, store = _sandbox(root)
    with pytest.raises(WorkspaceTamperedError):
        sb.run_code(
            "import os\n"
            "os.rename('handles', 'handles_real')\n"
            f"os.symlink({str(victim.parent)!r}, 'handles')\n")
    with pytest.raises(WorkspaceTamperedError):
        sb.run_code("1\n")                        # and every later run on this root
    assert victim.read_text() == "original"


def test_save_rejects_invalid_id_in_child(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code("from tether_sandbox import save\nsave('eu/sales', {'a': 1})\n")
    assert res.exit_code != 0
    assert "invalid handle id" in res.error and "letters, digits" in res.error
    assert store.manifest() == {}


def test_reused_id_is_reported(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    store.put("first", source="t", id="dup")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        "open(os.path.join(d, 'p.txt'), 'w').write('second')\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'dup', 'kind': 'text', 'file': 'p.txt'}) + '\\n')\n")
    assert "handle id already exists" in res.error and "dup" in res.error
    assert store.get("dup") == "first"


def test_emit_over_the_cap_is_rejected(tmp_path):
    sb, _ = _sandbox(tmp_path / "root", max_emit_bytes=100)
    res = sb.run_code("from tether_sandbox import emit\nemit('x' * 500)\n")
    assert res.result is None and "emit payload too large" in res.error


def test_stdout_is_capped(tmp_path):
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code("print('A' * 5_000_000)\n")
    assert len(res.stdout) < 9000
    assert "bytes of output omitted" in res.stdout
    assert len(json.dumps(res.result)) < 9000


def test_timeout_is_reported(tmp_path):
    root = tmp_path / "root"
    store = HandleStore(root)
    sb = LocalSubprocessSandbox(root=root, store=store,
                                config=SandboxConfig(backend="local", timeout_s=1))
    res = sb.run_code("import time\ntime.sleep(30)\n")
    assert res.killed_by == "timeout" and res.exit_code != 0
    assert "tether: killed (timeout)" in res.stderr


def test_publish_requests_travel_through_the_run_dir(tmp_path):
    root = tmp_path / "root"
    sb, _ = _sandbox(root)
    seen = []
    sb.publisher = lambda path, **kw: seen.append((path, kw)) or {"ok": True}
    (root / "outputs").mkdir()
    (root / "outputs" / "a.csv").write_text("x\n")
    res = sb.run_code("from tether_sandbox import publish\npublish('outputs/a.csv', name='A')\n")
    assert res.error is None, res.error
    assert seen == [("outputs/a.csv", {"name": "A", "description": None, "source": "run_python"})]
    assert res.published == [{"ok": True}]


def test_publish_is_unavailable_without_a_publisher(tmp_path):
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code("from tether_sandbox import publish\npublish('x')\n")
    assert res.exit_code != 0 and "publishing is not enabled" in res.error


def test_orphaned_launch_ingests_nothing(tmp_path):
    from tether.sandbox import _LaunchResult, _OrchestratedSandbox

    class _Fake(_OrchestratedSandbox):
        orphaned = True

        def _launch(self, ctx):
            (ctx.run_dir / "p.json").write_text(json.dumps({"x": 1}))
            (ctx.run_dir / "new_handles.jsonl").write_text(
                json.dumps({"id": "h9", "kind": "json", "file": "p.json"}) + "\n")
            (ctx.run_dir / "emit.json").write_text(json.dumps("emitted"))
            if self.orphaned:
                return _LaunchResult("", "", -1, killed_by="timeout", orphaned=True)
            return _LaunchResult("", "", 0)

    root = tmp_path / "root"
    store = HandleStore(root)
    sb = _Fake(root=root, store=store, config=SandboxConfig(backend="local"))
    res = sb.run_code("1\n")
    assert res.new_handles == [] and res.result is None
    assert "could not be stopped" in res.error and res.killed_by == "timeout"
    assert store.manifest() == {} and list((root / "handles").iterdir()) == []

    sb.orphaned = False                    # control: the same records are ingested normally
    res = sb.run_code("1\n")
    assert res.new_handles == ["h9"] and store.get("h9") == {"x": 1}
