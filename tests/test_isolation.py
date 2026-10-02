"""TETHER-1: isolation can be made mandatory; the container tier's guarantees are pinned."""

from pathlib import Path

import pytest

from tether import SandboxRuntimeUnavailable, Session, TetherConfig
from tether.config import SandboxConfig
from tether.handles import HandleStore
from tether.sandbox import _RunContext
from tether.sandbox_container import ContainerSandbox


def test_require_isolation_defaults_off():
    assert SandboxConfig().require_isolation is False


def test_local_backend_with_require_isolation_fails_at_session_create(tmp_path):
    cfg = TetherConfig(root_dir=tmp_path / "r",
                       sandbox=SandboxConfig(backend="local", require_isolation=True))
    with pytest.raises(SandboxRuntimeUnavailable) as ei:
        Session.create(cfg)
    msg = str(ei.value)
    assert "container" in msg and "require_isolation" in msg
    assert not (tmp_path / "r").exists()     # a root this call created is rolled back


def test_require_isolation_also_refuses_an_env_selected_local_backend(tmp_path, monkeypatch):
    """The env var picks the tier, so it is another way to ask for `local`.

    ``container`` is the shipped default, which makes ``require_isolation`` the stricter
    assertion on top of it: the refusal must not depend on *how* local was selected.
    """
    monkeypatch.setenv("TETHER_SANDBOX_BACKEND", "local")
    cfg = TetherConfig(root_dir=tmp_path / "r",
                       sandbox=SandboxConfig(require_isolation=True))
    assert cfg.sandbox.backend == "local"              # selected by the environment
    with pytest.raises(SandboxRuntimeUnavailable, match="require_isolation"):
        Session.create(cfg)
    assert not (tmp_path / "r").exists()


def test_local_backend_without_requirement_still_works(tmp_path):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r",
                                       sandbox=SandboxConfig(backend="local")))
    assert sess.sandbox.run_code("1 + 1").result == 2


def test_missing_container_runtime_raises_runtime_unavailable(tmp_path):
    cfg = TetherConfig(root_dir=tmp_path / "r", sandbox=SandboxConfig(
        backend="container", require_isolation=True, container_runtime="no-such-runtime-xyz"))
    with pytest.raises(SandboxRuntimeUnavailable):
        Session.create(cfg)


def test_runtime_unavailable_is_a_runtime_error():
    # Existing callers catching RuntimeError keep working.
    assert issubclass(SandboxRuntimeUnavailable, RuntimeError)


# --- container argv guarantees (no runtime needed) ---------------------------------------

def _argv(tmp_path, monkeypatch=None, **cfg):
    store = HandleStore(tmp_path)
    sb = ContainerSandbox(root=tmp_path, store=store,
                          config=SandboxConfig(backend="container", **cfg), runtime="podman")
    ctx = _RunContext(script_rel=".scripts/x.py", argv=[], root=tmp_path,
                      run_dir=tmp_path / ".tether" / "runs" / "tok", config=sb.config)
    return sb._build_run_argv(ctx, "img:abc", layer=None)


def _pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_container_argv_hardening_flags(tmp_path):
    argv = _argv(tmp_path)
    assert _pairs(argv, "--network") == ["none"]
    assert "--read-only" in argv
    assert _pairs(argv, "--cap-drop") == ["ALL"]
    assert "no-new-privileges" in _pairs(argv, "--security-opt")
    assert _pairs(argv, "--pids-limit") and _pairs(argv, "--memory") and _pairs(argv, "--cpus")
    assert "--privileged" not in argv


def test_container_mounts_only_root_inputs_and_runtime(tmp_path):
    # Exactly: the rw root, the ro runtime, the four parent-owned trees overlaid read-only
    # (inputs/ among them), and this run's exchange directory. Nothing else of the host.
    mounts = _pairs(_argv(tmp_path), "-v")
    assert mounts == [
        f"{tmp_path}:/workspace:rw",
        mounts[1],
        f"{tmp_path / 'handles'}:/workspace/handles:ro",
        f"{tmp_path / 'inputs'}:/workspace/inputs:ro",
        f"{tmp_path / '.scripts'}:/workspace/.scripts:ro",
        f"{tmp_path / '.tether'}:/workspace/.tether:ro",
        f"{tmp_path / '.tether' / 'runs' / 'tok'}:/tether-run:rw",
    ]
    assert mounts[1].endswith(":/runtime:ro")


def test_container_passes_no_host_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TETHER_TEST_HOST_SECRET", "s3cret")
    argv = _argv(tmp_path)
    assert "--env-host" not in argv and "--env-file" not in argv
    envs = _pairs(argv, "-e")
    keys = {e.split("=", 1)[0] for e in envs}
    assert all("=" in e for e in envs)     # `-e KEY` without a value would inherit from the host
    assert keys - {"PYTHONPATH", "HOME", "TMPDIR", "PATH"} == {
        k for k in keys if k.startswith("TETHER_")}
    assert not any("s3cret" in a for a in argv)


def test_container_sandbox_docs_say_local_is_not_a_boundary():
    doc = " ".join((SandboxConfig.__doc__ or "").split())
    assert "not a security boundary" in doc
    readme = " ".join(
        (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8").split())
    assert "not a security boundary" in readme
