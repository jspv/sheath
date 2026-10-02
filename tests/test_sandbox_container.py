from tether.config import SandboxConfig
from tether.sandbox import _RunContext
from tether.sandbox_container import ContainerSandbox


def _ctx(tmp_path, config, publish_enabled=False):
    return _RunContext(
        script_rel=".scripts/inline_x.py", argv=["EU", "2025"], root=tmp_path,
        run_dir=tmp_path / ".tether" / "runs" / "tok", config=config,
        publish_enabled=publish_enabled,
    )


def _sandbox(tmp_path, config):
    # inject a fake runtime so construction needs no podman/docker installed
    return ContainerSandbox(root=tmp_path, store=None, config=config, runtime="podman")


def test_run_argv_maps_userns_for_podman_only(tmp_path):
    cfg = SandboxConfig(backend="container")
    # podman: rootless maps host user -> container-root, so keep-id is needed for the bind mount
    podman = ContainerSandbox(root=tmp_path, store=None, config=cfg, runtime="podman")
    assert "--userns=keep-id" in podman._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)
    # docker rootless uses a different model and rejects keep-id
    docker = ContainerSandbox(root=tmp_path, store=None, config=cfg, runtime="docker")
    assert "--userns=keep-id" not in docker._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)


def test_run_argv_blocks_network_by_default(tmp_path):
    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "tether-sandbox:abc", layer=None)
    assert argv[:2] == ["podman", "run"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv and "--cap-drop" in argv
    assert "no-new-privileges" in " ".join(argv)


def test_run_argv_enables_network_when_configured(tmp_path):
    cfg = SandboxConfig(backend="container", network=True)
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "tether-sandbox:abc", layer=None)
    assert "--network" not in argv                          # network not blocked


def test_run_argv_mounts_root_and_runtime_and_translates_paths(tmp_path):
    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "tether-sandbox:abc", layer=None)
    joined = " ".join(argv)
    assert f"{tmp_path}:/workspace:rw" in joined            # root mount
    assert ":/runtime:ro" in joined                         # runtime mount (read-only)
    assert "TETHER_ROOT=/workspace" in joined              # path translated into the container
    assert "TETHER_RUN_DIR=/tether-run" in joined
    assert "PYTHONPATH=/runtime" in joined
    # the command ends by running the runner with the relative script path + user args
    assert argv[-5:] == ["python", "/runtime/_runner.py", ".scripts/inline_x.py", "EU", "2025"]


def test_run_argv_mounts_layer_and_extends_pythonpath(tmp_path):
    cfg = SandboxConfig(backend="container", pip_packages=("rich",))
    sb = _sandbox(tmp_path, cfg)
    layer = tmp_path / "layer"
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "tether-sandbox:abc", layer=layer)
    joined = " ".join(argv)
    assert f"{layer}:/pkgs:ro" in joined
    assert "PYTHONPATH=/runtime:/pkgs" in joined


def test_run_argv_applies_resource_limits(tmp_path):
    cfg = SandboxConfig(backend="container", max_memory_mb=512, max_cpus=1.5)
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "tether-sandbox:abc", layer=None)
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "512m"
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "1.5"
    assert "--pids-limit" in argv


def _mounts(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def test_run_argv_mounts_protected_dirs_read_only(tmp_path):
    cfg = SandboxConfig(backend="container")
    argv = _sandbox(tmp_path, cfg)._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)
    mounts = _mounts(argv)
    assert f"{tmp_path}:/workspace:rw" in mounts
    for name in ("handles", "inputs", ".scripts", ".tether"):
        assert f"{tmp_path / name}:/workspace/{name}:ro" in mounts
    assert f"{tmp_path / '.tether' / 'runs' / 'tok'}:/tether-run:rw" in mounts
    # the workspace mount comes first so the read-only mounts overlay it
    assert mounts.index(f"{tmp_path}:/workspace:rw") < mounts.index(
        f"{tmp_path / 'handles'}:/workspace/handles:ro")


def test_run_argv_passes_the_run_dir_protocol_env(tmp_path):
    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "TETHER_ROOT=/workspace" in envs and "TETHER_RUN_DIR=/tether-run" in envs
    assert not any(e.startswith(("TETHER_EMIT", "TETHER_REGISTRY", "TETHER_NEW_HANDLES",
                                 "TETHER_PUBLISH")) for e in envs)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg, publish_enabled=True), "img:abc", layer=None)
    assert "TETHER_PUBLISH_ENABLED=1" in [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]


def test_timed_out_container_that_cannot_be_removed_is_orphaned(tmp_path, monkeypatch):
    from tether import sandbox_container
    from tether.procio import BoundedRun

    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    monkeypatch.setattr("tether.container_runtime.ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(sandbox_container, "run_bounded",
                        lambda *a, **k: BoundedRun("", "", -9, True))
    monkeypatch.setattr(sb, "_remove_container", lambda name: "\ntether: WARNING: stuck")
    launched = sb._launch(_ctx(tmp_path, cfg))
    assert launched.orphaned and launched.killed_by == "timeout"

    monkeypatch.setattr(sb, "_remove_container", lambda name: "")
    assert not sb._launch(_ctx(tmp_path, cfg)).orphaned
