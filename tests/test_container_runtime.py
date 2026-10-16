import pytest

from tether.config import SandboxConfig
from tether.container_runtime import (
    detect_runtime,
    ensure_image,
    ensure_layer,
    image_tag,
    layer_dir,
)


def test_detect_prefers_podman_then_docker():
    assert detect_runtime(None, which=lambda c: c if c == "podman" else None) == "podman"
    assert detect_runtime(None, which=lambda c: c if c == "docker" else None) == "docker"


def test_detect_honors_override():
    assert detect_runtime("docker", which=lambda c: "/usr/bin/docker") == "docker"


def test_detect_raises_when_none_found():
    with pytest.raises(RuntimeError, match="no container runtime"):
        detect_runtime(None, which=lambda c: None)


def test_detect_raises_when_override_missing():
    with pytest.raises(RuntimeError, match="not found"):
        detect_runtime("nope", which=lambda c: None)


def test_image_tag_is_stable_and_depends_on_preinstalled():
    a = image_tag(("pandas", "numpy"))
    assert a == image_tag(("numpy", "pandas"))        # order-independent
    assert a.startswith("tether-sandbox:")
    assert a != image_tag(("pandas",))                # different set -> different tag


def test_layer_dir_keys_on_packages(tmp_path):
    cfg1 = SandboxConfig(pip_packages=("rich",))
    cfg2 = SandboxConfig(pip_packages=("rich", "tabulate"))
    d1 = layer_dir(cfg1, base=tmp_path)
    assert d1 == layer_dir(cfg1, base=tmp_path)        # stable
    assert d1 != layer_dir(cfg2, base=tmp_path)        # different set -> different dir
    assert d1.parent == tmp_path


def test_ensure_image_builds_when_absent():
    calls = []

    class _Proc:
        def __init__(self, code, out="", err=""):
            self.returncode, self.stdout, self.stderr = code, out, err

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return _Proc(1)                            # not present -> triggers build
        return _Proc(0)

    ensure_image("podman", "tether-sandbox:abc", SandboxConfig(), run=fake_run)
    assert any(a[1] == "build" and "-t" in a and "tether-sandbox:abc" in a for a in calls)


def test_ensure_image_skips_build_when_present():
    class _Proc:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, **kw):
        assert argv[1] != "build", "should not build when image exists"
        return _Proc()

    ensure_image("podman", "tether-sandbox:abc", SandboxConfig(), run=fake_run)


def test_ensure_layer_provisions_once_then_serves_from_sentinel(tmp_path):
    calls = []

    class _Proc:
        returncode, stdout, stderr = 0, "", ""   # image present + pip install succeed

    def fake_run(argv, **kw):
        calls.append(argv)
        return _Proc()

    cfg = SandboxConfig(pip_packages=("six",))
    first = ensure_layer("podman", cfg, base=tmp_path, run=fake_run)
    assert any("pip" in a for a in calls)               # provisioned (pip install ran)
    n = len(calls)
    second = ensure_layer("podman", cfg, base=tmp_path, run=fake_run)
    assert second == first
    assert len(calls) == n                              # sentinel hit -> nothing re-run


def test_ensure_layer_reprovisions_if_sentinel_missing(tmp_path):
    # A non-empty but incomplete dir (provision crashed) must NOT be served as cached.
    cfg = SandboxConfig(pip_packages=("six",))
    target = layer_dir(cfg, base=tmp_path)
    target.mkdir(parents=True)
    (target / "partial").write_text("x")                # non-empty, but no .complete sentinel
    calls = []

    class _Proc:
        returncode, stdout, stderr = 0, "", ""

    ensure_layer("podman", cfg, base=tmp_path, run=lambda a, **k: calls.append(a) or _Proc())
    assert any("pip" in a for a in calls)               # re-provisioned despite non-empty dir


# --- default backend guard / liveness probe -------------------------------------------

import subprocess  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from tether.container_runtime import require_usable_runtime  # noqa: E402
from tether.handles import HandleStore  # noqa: E402
from tether.sandbox import SandboxRuntimeUnavailable  # noqa: E402
from tether.session import _build_sandbox  # noqa: E402


def test_missing_runtime_raises_and_names_the_opt_out(tmp_path, monkeypatch):
    def no_runtime(override, which=None):
        raise RuntimeError("no container runtime found (looked for podman, docker)")

    # ContainerSandbox.__init__ does `from .container_runtime import detect_runtime` at call
    # time, so patching the module attribute is what takes effect.
    monkeypatch.setattr("tether.container_runtime.detect_runtime", no_runtime)
    store = HandleStore(tmp_path / "r")

    with pytest.raises(SandboxRuntimeUnavailable) as excinfo:
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="container"))

    message = str(excinfo.value)
    assert 'backend = "local"' in message       # tells the user exactly how to proceed
    assert "no isolation" in message


def test_local_backend_needs_no_runtime(tmp_path):
    store = HandleStore(tmp_path / "r")
    sandbox = _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="local"))
    assert type(sandbox).__name__ == "LocalSubprocessSandbox"


def _present(monkeypatch, name="podman"):
    monkeypatch.setattr("tether.container_runtime.detect_runtime",
                        lambda override, which=None: name)


def test_usable_runtime_returned_on_zero_exit(monkeypatch):
    _present(monkeypatch)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr=b"")

    assert require_usable_runtime(None, run=run) == "podman"
    assert calls == [["podman", "info"]]


def test_installed_but_not_responding_raises_with_stderr_hint(monkeypatch):
    _present(monkeypatch)

    def run(cmd, **kw):
        return SimpleNamespace(returncode=125, stderr=b"Cannot connect to Podman\nmore\n")

    with pytest.raises(RuntimeError) as excinfo:
        require_usable_runtime(None, run=run)
    assert "Cannot connect to Podman" in str(excinfo.value)
    assert "podman machine start" in str(excinfo.value)


def test_probe_oserror_raises(monkeypatch):
    _present(monkeypatch)

    def run(cmd, **kw):
        raise OSError("exec format error")

    with pytest.raises(RuntimeError, match="not usable"):
        require_usable_runtime(None, run=run)


def test_probe_timeout_raises(monkeypatch):
    _present(monkeypatch)

    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 30)

    with pytest.raises(RuntimeError, match="not usable"):
        require_usable_runtime(None, run=run)


def test_build_sandbox_surfaces_dead_runtime_as_unavailable(tmp_path, monkeypatch):
    """Installed-but-not-started: detection passes, the probe fails, the user is told how out."""
    monkeypatch.setattr("tether.container_runtime.detect_runtime",
                        lambda override, which=None: "podman")
    # `run=subprocess.run` is bound at def time, so patching subprocess.run would be a no-op;
    # wrap the real probe with a fake runner instead.
    real = require_usable_runtime
    monkeypatch.setattr(
        "tether.container_runtime.require_usable_runtime",
        lambda override: real(override, run=lambda cmd, **kw: SimpleNamespace(
            returncode=125, stderr=b"Cannot connect to Podman")),
    )
    store = HandleStore(tmp_path / "r")

    with pytest.raises(SandboxRuntimeUnavailable) as excinfo:
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="container"))
    assert "Cannot connect to Podman" in str(excinfo.value)
    assert 'backend = "local"' in str(excinfo.value)


# --- fail closed on unknown backend; local tier leaves a trace -------------------------

@pytest.mark.parametrize("bad", ["Container", "CONTAINER", "contianer", "podman", "docker", ""])
def test_unknown_backend_fails_closed(tmp_path, bad):
    store = HandleStore(tmp_path / "r")
    with pytest.raises(ValueError) as excinfo:
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend=bad))
    message = str(excinfo.value)
    assert "'container'" in message and "'local'" in message
    assert "container_runtime" in message       # podman/docker are not tier names


def test_valid_backends_still_build(tmp_path, monkeypatch):
    monkeypatch.setattr("tether.container_runtime.detect_runtime",
                        lambda override, which=None: "podman")
    store = HandleStore(tmp_path / "r")
    assert type(_build_sandbox(tmp_path / "r", store,
                               SandboxConfig(backend="local"))).__name__ == "LocalSubprocessSandbox"


def test_local_tier_emits_no_isolation_warning(tmp_path):
    from tether import NoSandboxIsolationWarning

    store = HandleStore(tmp_path / "r")
    with pytest.warns(NoSandboxIsolationWarning, match="NO isolation"):
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="local"))


def test_constructor_runtimeerror_is_not_relabelled_as_a_missing_runtime(tmp_path, monkeypatch):
    """Only the probe's RuntimeErrors mean 'no usable runtime'. A future RuntimeError from
    ContainerSandbox.__init__ must not send the user off to install something they have."""
    monkeypatch.setattr("tether.container_runtime.require_usable_runtime",
                        lambda override: "podman")

    def boom(**kw):
        raise RuntimeError("bind mount setup failed")

    monkeypatch.setattr("tether.sandbox_container.ContainerSandbox", boom)
    store = HandleStore(tmp_path / "r")

    with pytest.raises(RuntimeError) as excinfo:
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="container"))
    assert not isinstance(excinfo.value, SandboxRuntimeUnavailable)
    assert "bind mount setup failed" in str(excinfo.value)


def test_a_successful_probe_is_cached_per_process(monkeypatch):
    """`<runtime> info` is a subprocess with a 30s timeout, run on the event loop once per
    Session.create -- i.e. once per conversation. A multi-conversation host pays it once."""
    _present(monkeypatch)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr=b"")

    assert require_usable_runtime(None, run=run) == "podman"
    assert require_usable_runtime(None, run=run) == "podman"
    assert len(calls) == 1


def test_the_cache_is_keyed_on_the_override(monkeypatch):
    monkeypatch.setattr("tether.container_runtime.detect_runtime",
                        lambda override, which=None: override or "podman")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[0])
        return SimpleNamespace(returncode=0, stderr=b"")

    require_usable_runtime(None, run=run)
    require_usable_runtime("docker", run=run)
    require_usable_runtime("docker", run=run)
    assert calls == ["podman", "docker"]


def test_a_failed_probe_is_not_cached(monkeypatch):
    """Starting the VM must be enough; the user should not have to restart the process."""
    _present(monkeypatch)
    state = {"up": False}

    def run(cmd, **kw):
        return SimpleNamespace(returncode=0 if state["up"] else 125, stderr=b"not running")

    with pytest.raises(RuntimeError):
        require_usable_runtime(None, run=run)
    state["up"] = True                                  # the user starts their VM
    assert require_usable_runtime(None, run=run) == "podman"


def test_image_build_targets_the_host_python_version():
    import sys
    import types
    from tether.config import SandboxConfig
    from tether.container_runtime import ensure_image

    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        missing = argv[1:3] == ["image", "inspect"]
        return types.SimpleNamespace(returncode=1 if missing else 0, stdout="", stderr="")

    ensure_image("podman", "tether-sandbox:x",
                 SandboxConfig(backend="container", build_on_demand=True), run)
    build = calls[-1]
    assert build[:2] == ["podman", "build"]
    want = f"PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}"
    assert want in build and build[build.index(want) - 1] == "--build-arg"


def test_containerfile_takes_the_python_version_as_an_argument():
    from tether.container_runtime import _CONTAINERFILE
    text = _CONTAINERFILE.read_text()
    assert "ARG PYTHON_VERSION=3.12" in text
    assert "FROM python:${PYTHON_VERSION}-slim" in text
