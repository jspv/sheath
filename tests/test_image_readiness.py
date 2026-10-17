"""TETHER-2: no hidden builds, bounded build time, non-mutating preflight."""

import functools
import subprocess
import time

import pytest

from sheath import PreflightReport, SheathConfig, sandbox_preflight
from sheath.config import SandboxConfig
from sheath.container_runtime import (
    SandboxImageError,
    SandboxImageMissing,
    _build_sandbox_main,
    ensure_image,
    ensure_layer,
    image_tag,
    layer_dir,
)


class _Proc:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def _missing_image_run(calls):
    def run(argv, **kw):
        calls.append((argv, kw))
        if argv[1:3] == ["image", "inspect"]:
            return _Proc(1, err="no such image")
        return _Proc(0)
    return run


# --- build_on_demand resolution ---------------------------------------------------------

def test_build_on_demand_auto_follows_require_isolation():
    # Unchanged by the container-by-default flip: `None` still means "build unless
    # isolation is required", so an out-of-the-box container run builds its image on first
    # use, and a host that demands isolation is still required to pre-build.
    assert SandboxConfig().build_on_demand is None
    assert SandboxConfig().effective_build_on_demand is True            # 0.1 behavior kept
    assert SandboxConfig(require_isolation=True).effective_build_on_demand is False
    assert SandboxConfig(require_isolation=True,
                         build_on_demand=True).effective_build_on_demand is True
    assert SandboxConfig(build_on_demand=False).effective_build_on_demand is False


# --- ensure_image ------------------------------------------------------------------------

def test_missing_image_with_builds_disabled_raises_fast_without_building():
    calls = []
    cfg = SandboxConfig(build_on_demand=False, preinstalled=("pandas",))
    tag = image_tag(cfg.preinstalled)
    t0 = time.monotonic()
    with pytest.raises(SandboxImageMissing) as ei:
        ensure_image("podman", tag, cfg, run=_missing_image_run(calls))
    assert time.monotonic() - t0 < 1.0
    assert not any(a[1] == "build" for a, _ in calls)
    msg = str(ei.value)
    assert tag in msg and "sheath-build-sandbox" in msg and "--preinstalled pandas" in msg


def test_image_inspect_is_time_bounded():
    calls = []
    ensure_image("podman", "t:1", SandboxConfig(), run=lambda a, **k: calls.append(k) or _Proc(0))
    assert calls[0].get("timeout")


def test_build_runs_with_timeout_and_raises_on_expiry():
    def run(argv, **kw):
        if argv[1:3] == ["image", "inspect"]:
            return _Proc(1)
        assert kw.get("timeout") == 5.0
        raise subprocess.TimeoutExpired(argv, kw["timeout"])   # a build that never returns

    with pytest.raises(SandboxImageError, match="timed out"):
        ensure_image("podman", "t:1", SandboxConfig(build_timeout_s=5.0), run=run)


def test_inspect_timeout_is_reported_not_hung():
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout") or 0)

    with pytest.raises(SandboxImageError, match="did not respond"):
        ensure_image("podman", "t:1", SandboxConfig(build_on_demand=False), run=run)


# --- ensure_layer ------------------------------------------------------------------------

def test_unprovisioned_layer_with_builds_disabled_raises(tmp_path):
    calls = []
    cfg = SandboxConfig(pip_packages=("six",), build_on_demand=False)
    with pytest.raises(SandboxImageMissing, match="--pip six"):
        ensure_layer("podman", cfg, base=tmp_path, run=lambda a, **k: calls.append(a) or _Proc(0))
    assert not any("pip" in a for a in calls)


def test_layer_provisioning_times_out(tmp_path):
    def run(argv, **kw):
        if "pip" in argv:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return _Proc(0)

    cfg = SandboxConfig(pip_packages=("six",), build_timeout_s=3.0)
    with pytest.raises(SandboxImageError, match="timed out"):
        ensure_layer("podman", cfg, base=tmp_path, run=run)
    assert not (layer_dir(cfg, base=tmp_path).parent / (layer_dir(cfg, base=tmp_path).name
                                                         + ".complete")).exists()


# --- run_python surfaces the error --------------------------------------------------------

def test_run_python_with_missing_image_errors_quickly(tmp_path, monkeypatch):
    import sheath.container_runtime as cr
    from sheath.handles import HandleStore
    from sheath.sandbox_container import ContainerSandbox

    calls = []
    # ensure_image binds subprocess.run as a default arg, so inject the fake through it.
    monkeypatch.setattr(cr, "ensure_image",
                        functools.partial(cr.ensure_image, run=_missing_image_run(calls)))
    sb = ContainerSandbox(root=tmp_path, store=HandleStore(tmp_path), runtime="podman",
                          config=SandboxConfig(backend="container", build_on_demand=False))
    t0 = time.monotonic()
    with pytest.raises(SandboxImageMissing):
        sb.run_code("1")
    assert time.monotonic() - t0 < 1.0
    assert not any(a[1] == "build" for a, _ in calls)


# --- preflight ---------------------------------------------------------------------------

def _no_build(calls):
    def run(argv, **kw):
        calls.append(argv)
        assert argv[1] not in ("build", "run"), "preflight must never build or run"
        return _Proc(0)
    return run


def test_preflight_ok_when_runtime_and_image_present():
    calls = []
    rep = sandbox_preflight(SandboxConfig(backend="container"),
                            which=lambda c: f"/bin/{c}", run=_no_build(calls))
    assert isinstance(rep, PreflightReport)
    assert rep.ok and rep.problems == [] and rep.runtime == "podman"
    assert rep.image_tag == image_tag(SandboxConfig().preinstalled)


def test_preflight_accepts_sheath_config():
    rep = sandbox_preflight(SheathConfig(sandbox=SandboxConfig(backend="container")),
                            which=lambda c: f"/bin/{c}", run=_no_build([]))
    assert rep.ok


def test_preflight_reports_missing_runtime():
    rep = sandbox_preflight(SandboxConfig(backend="container"), which=lambda c: None,
                            run=_no_build([]))
    assert not rep.ok and rep.runtime is None
    assert any("runtime" in p for p in rep.problems)


def test_preflight_reports_missing_image_and_never_builds():
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        assert argv[1] != "build"
        assert kw.get("timeout")
        return _Proc(1, err="Error: no such image")

    rep = sandbox_preflight(SandboxConfig(backend="container"),
                            which=lambda c: f"/bin/{c}", run=run)
    assert not rep.ok
    assert any(rep.image_tag in p and "sheath-build-sandbox" in p for p in rep.problems)


def test_preflight_reports_hung_daemon():
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    rep = sandbox_preflight(SandboxConfig(backend="container"),
                            which=lambda c: f"/bin/{c}", run=run)
    assert not rep.ok and any("respond" in p for p in rep.problems)


def test_preflight_reports_unprovisioned_layer(tmp_path):
    rep = sandbox_preflight(SandboxConfig(backend="container", pip_packages=("six",)),
                            which=lambda c: f"/bin/{c}", run=_no_build([]), layer_base=tmp_path)
    assert not rep.ok and any("pip_packages" in p for p in rep.problems)


def test_preflight_local_backend():
    # Explicit: the shipped default is `container`, and the suite's autouse fixture pins
    # `local` through the environment -- neither may decide what this test checks.
    assert sandbox_preflight(SandboxConfig(backend="local")).ok
    rep = sandbox_preflight(SandboxConfig(backend="local", require_isolation=True))
    assert not rep.ok and any("require_isolation" in p for p in rep.problems)


# --- sheath-build-sandbox CLI --------------------------------------------------------------

def test_build_cli_uses_host_inputs(monkeypatch, capsys):
    import sheath.container_runtime as cr

    seen = {}
    monkeypatch.setattr(cr, "detect_runtime", lambda override, **k: override or "podman")
    monkeypatch.setattr(cr, "ensure_image",
                        lambda rt, tag, cfg, **k: seen.update(tag=tag, cfg=cfg))
    monkeypatch.setattr(cr, "ensure_layer",
                        lambda rt, cfg, **k: seen.update(layer=cfg.pip_packages))
    _build_sandbox_main(["--preinstalled", "pandas", "numpy", "--pip", "six",
                         "--runtime", "docker", "--timeout", "60"])
    assert seen["tag"] == image_tag(("pandas", "numpy"))
    assert seen["cfg"].build_timeout_s == 60.0 and seen["cfg"].effective_build_on_demand
    assert seen["layer"] == ("six",)
    assert seen["tag"] in capsys.readouterr().out


def test_build_cli_check_mode_exits_nonzero_on_problems(monkeypatch, capsys):
    import sheath.container_runtime as cr

    monkeypatch.setattr(cr, "sandbox_preflight", lambda cfg, **k: PreflightReport(
        ok=False, backend="container", runtime=None, image_tag="t", problems=["nope"]))
    with pytest.raises(SystemExit) as ei:
        _build_sandbox_main(["--check"])
    assert ei.value.code == 1
    assert "nope" in capsys.readouterr().out


def test_layer_build_shares_one_deadline(tmp_path, monkeypatch):
    import sheath.container_runtime as cr

    clock = [0.0]
    monkeypatch.setattr(cr.time, "monotonic", lambda: clock[0])
    seen = {}

    def run(argv, **kw):
        if argv[1:3] == ["image", "inspect"]:
            return _Proc(1)
        if argv[1] == "build":
            seen["build"] = kw["timeout"]
            clock[0] += 60.0                      # the build took 60 s of the budget
            return _Proc(0)
        if "pip" in argv:
            seen["pip"] = kw["timeout"]
        return _Proc(0)

    ensure_layer("podman", SandboxConfig(pip_packages=("six",), build_timeout_s=100.0),
                 base=tmp_path, run=run)
    assert seen["build"] == 100.0 and seen["pip"] == pytest.approx(40.0)


def test_exhausted_deadline_raises_before_pip(tmp_path, monkeypatch):
    import sheath.container_runtime as cr

    clock = [0.0]
    monkeypatch.setattr(cr.time, "monotonic", lambda: clock[0])
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return _Proc(1)
        if argv[1] == "build":
            clock[0] += 150.0
        return _Proc(0)

    with pytest.raises(SandboxImageError, match="timed out"):
        ensure_layer("podman", SandboxConfig(pip_packages=("six",), build_timeout_s=100.0),
                     base=tmp_path, run=run)
    assert not any("pip" in a for a in calls)


def test_layer_timeout_removes_named_container(tmp_path):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if "pip" in argv:
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return _Proc(0)

    with pytest.raises(SandboxImageError):
        ensure_layer("podman", SandboxConfig(pip_packages=("six",)), base=tmp_path, run=run)
    pip_argv = next(a for a in calls if "pip" in a)
    name = pip_argv[pip_argv.index("--name") + 1]
    assert name.startswith("sheath-layer-")
    assert ["podman", "rm", "-f", name] in calls
