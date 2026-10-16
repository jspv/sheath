"""Container runtime helpers: detect podman/docker, build the sandbox image, provision a
package layer. Driven via the runtime CLI (subprocess) -- no Python container SDK dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .config import SandboxConfig, TetherConfig

_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
_CONTAINERFILE = _RUNTIME_DIR / "Containerfile"
_RM_TIMEOUT_S = 30.0
_INSPECT_TIMEOUT_S = 10.0   # `image inspect` must never hang a check, even on a wedged daemon


class SandboxImageError(RuntimeError):
    """The sandbox image or package layer could not be made ready (failed or timed out)."""


class SandboxImageMissing(SandboxImageError):
    """The sandbox image or package layer is absent and building on demand is disabled."""


def detect_runtime(override: str | None, which: Callable[[str], str | None] = shutil.which) -> str:
    """Return the container runtime binary name. Prefers ``override``, then podman, then docker."""
    if override:
        if which(override):
            return override
        raise RuntimeError(f"container runtime {override!r} not found on PATH")
    for candidate in ("podman", "docker"):
        if which(candidate):
            return candidate
    # Facts only. The caller that selected the container backend owns the remediation --
    # see Session._build_sandbox, which wraps this. Repeating "install podman or docker, or
    # set backend='local'" here made the user read the same advice twice, in two different
    # spellings, in one error.
    raise RuntimeError("no container runtime found: neither podman nor docker is on PATH")


_usable_runtime_cache: dict[str | None, str] = {}


def require_usable_runtime(override: str | None, run: Callable = subprocess.run) -> str:
    """Return a runtime that is actually usable, else raise RuntimeError.

    ``detect_runtime`` only checks PATH. A machine with podman installed but its VM not
    started passes that check and then fails at first use with an opaque image-build error.

    **Successes are cached per process, keyed on the override; failures never are.** The
    probe shells out to ``<runtime> info`` with a 30 s timeout, and since the container
    backend became the default this runs on the event loop once per ``Session.create`` --
    that is once per *conversation* on a multi-conversation host, which would pay for the
    same answer over and over. A runtime that answered once is not going to stop being
    installed. The reverse is not true: a user who starts their VM after a failure must get
    a working harness on the next try, not be told to restart the process.
    """
    if override in _usable_runtime_cache:
        return _usable_runtime_cache[override]
    runtime = detect_runtime(override)
    try:
        proc = run([runtime, "info"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"container runtime {runtime!r} is installed but not usable: {e}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode(errors="replace").strip().splitlines()
        hint = detail[0] if detail else f"exit {proc.returncode}"
        raise RuntimeError(
            f"container runtime {runtime!r} is installed but not responding ({hint}). "
            f"On macOS this usually means the VM is not started -- try `{runtime} machine start`."
        )
    _usable_runtime_cache[override] = runtime
    return runtime


def _py_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def image_tag(preinstalled: tuple[str, ...]) -> str:
    """Stable image tag keyed by the Python version + the (order-independent) preinstalled set."""
    digest = hashlib.sha256((_py_tag() + "|" + ",".join(sorted(preinstalled))).encode()).hexdigest()
    return f"tether-sandbox:{digest[:12]}"


def image_exists(runtime: str, tag: str, run: Callable = subprocess.run) -> bool:
    """Whether ``tag`` exists locally. Time-bounded; raises SandboxImageError if the runtime
    does not answer within ``_INSPECT_TIMEOUT_S``."""
    # `image inspect` works on both podman and docker (unlike podman-only `image exists`).
    try:
        proc = run([runtime, "image", "inspect", tag], capture_output=True,
                   timeout=_INSPECT_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise SandboxImageError(
            f"container runtime {runtime!r} did not respond to `image inspect` within "
            f"{_INSPECT_TIMEOUT_S:.0f}s; is the daemon/machine running?") from e
    return proc.returncode == 0


def build_command_hint(config: SandboxConfig) -> str:
    """The `tether-build-sandbox` invocation that pre-builds exactly what ``config`` asks for."""
    parts = ["tether-build-sandbox", "--preinstalled", *config.preinstalled]
    if config.pip_packages:
        parts += ["--pip", *config.pip_packages]
    if config.container_runtime:
        parts += ["--runtime", config.container_runtime]
    return shlex.join(parts)


def _remaining(deadline: float, what: str) -> float:
    """Seconds left before ``deadline``; raises once the shared build budget is spent."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise SandboxImageError(f"{what} timed out (build_timeout_s exhausted)")
    return left


def ensure_image(runtime: str, tag: str, config: SandboxConfig,
                 run: Callable = subprocess.run, *, timeout_s: float | None = None) -> None:
    """Make sure the sandbox image exists.

    If it is missing and ``config.effective_build_on_demand`` is false, raise
    ``SandboxImageMissing`` immediately; otherwise build it, bounded by
    ``config.build_timeout_s``. Raises ``SandboxImageError`` on a failed or timed-out build.
    """
    if image_exists(runtime, tag, run):
        return
    if not config.effective_build_on_demand:
        raise SandboxImageMissing(
            f"sandbox image {tag} is not present and build_on_demand is disabled. "
            f"Pre-build it with: {build_command_hint(config)}")
    build = [runtime, "build", "-t", tag,
             "--build-arg", f"PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
             "--build-arg", f"PREINSTALLED={' '.join(config.preinstalled)}",
             "-f", str(_CONTAINERFILE), str(_RUNTIME_DIR)]
    budget = config.build_timeout_s if timeout_s is None else timeout_s
    try:
        proc = run(build, capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired as e:
        raise SandboxImageError(
            f"building sandbox image {tag} timed out after {budget:.0f}s "
            f"(can the runtime reach the base-image registry?)") from e
    if proc.returncode != 0:
        raise SandboxImageError(f"failed to build sandbox image {tag}:\n{proc.stderr}")


def layer_dir(config: SandboxConfig, base: Path | None = None) -> Path:
    """Host cache dir for the provisioned package layer, keyed by packages + Python version."""
    base = base or (Path.home() / ".tether" / "pkgcache")
    digest = hashlib.sha256(
        (_py_tag() + "|" + ",".join(sorted(config.pip_packages))).encode()
    ).hexdigest()
    return base / digest[:12]


def ensure_layer(runtime: str, config: SandboxConfig, base: Path | None = None,
                 run: Callable = subprocess.run) -> Path:
    """Provision ``config.pip_packages`` into a mounted layer (network ON, provisioning only).

    Cached by package set; provisions once. A sibling ``<dir>.complete`` sentinel marks success,
    so a crashed/partial provision is re-run rather than served as a broken layer. Returns the
    host layer dir.
    """
    target = layer_dir(config, base)
    sentinel = _layer_sentinel(target)
    if sentinel.exists():
        return target
    if not config.effective_build_on_demand:
        raise SandboxImageMissing(
            f"pip_packages layer {target} is not provisioned and build_on_demand is disabled. "
            f"Pre-build it with: {build_command_hint(config)}")
    target.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + config.build_timeout_s   # one budget for build + install
    tag = image_tag(config.preinstalled)
    ensure_image(runtime, tag, config, run,
                 timeout_s=_remaining(deadline, f"building sandbox image {tag}"))
    name = f"tether-layer-{secrets.token_hex(8)}"
    cmd = [runtime, "run", "--rm", "--name", name]
    # See ContainerSandbox._build_run_argv: rootless podman maps the host user to container-root,
    # so the host-owned /layer bind mount is unwritable to the hardening --user. keep-id maps the
    # host uid through so --user owns the mount. (podman-only; docker rootless rejects it.)
    if runtime == "podman":
        cmd += ["--userns=keep-id"]
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{target}:/layer:rw", tag,
            "pip", "install", "--no-cache-dir", "--target", "/layer", *config.pip_packages]
    try:
        proc = run(cmd, capture_output=True, text=True,
                   timeout=_remaining(deadline, f"provisioning pip_packages into {target}"))
    except subprocess.TimeoutExpired as e:
        try:   # killing the CLI does not stop the container; remove it by name
            run([runtime, "rm", "-f", name], capture_output=True, timeout=_RM_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        raise SandboxImageError(
            f"provisioning pip_packages into {target} timed out after "
            f"{config.build_timeout_s:.0f}s") from e
    if proc.returncode != 0:
        raise SandboxImageError(f"failed to provision pip_packages into {target}:\n{proc.stderr}")
    sentinel.write_text("ok", encoding="utf-8")   # only after a successful install
    return target


def _layer_sentinel(target: Path) -> Path:
    return target.parent / f"{target.name}.complete"


@dataclass(frozen=True)
class PreflightReport:
    """Result of ``sandbox_preflight``: ``ok`` is true only when ``problems`` is empty."""
    ok: bool
    backend: str
    runtime: str | None
    image_tag: str | None
    problems: list[str] = field(default_factory=list)


def sandbox_preflight(config: SandboxConfig | TetherConfig, *,
                      which: Callable[[str], str | None] = shutil.which,
                      run: Callable = subprocess.run,
                      layer_base: Path | None = None) -> PreflightReport:
    """Check, without building or running anything, that sandboxed code can run as configured.

    Verifies the isolation requirement against the backend, that the container runtime is on
    PATH, that the image exists (bounded ``image inspect``), and that the ``pip_packages``
    layer is provisioned. Intended for host startup / catalog load.
    """
    cfg = config.sandbox if isinstance(config, TetherConfig) else config
    problems: list[str] = []
    if cfg.backend != "container":
        if cfg.require_isolation:
            problems.append(f"require_isolation=True but backend={cfg.backend!r}; the local "
                            "tier is not a security boundary (use backend='container')")
        return PreflightReport(ok=not problems, backend=cfg.backend, runtime=None,
                               image_tag=None, problems=problems)

    tag = image_tag(cfg.preinstalled)
    try:
        runtime = detect_runtime(cfg.container_runtime, which=which)
    except RuntimeError as e:
        problems.append(f"container runtime unavailable: {e}")
        return PreflightReport(ok=False, backend=cfg.backend, runtime=None, image_tag=tag,
                               problems=problems)
    try:
        if not image_exists(runtime, tag, run):
            problems.append(f"sandbox image {tag} not found (or the runtime is unreachable); "
                            f"build it with: {build_command_hint(cfg)}")
    except SandboxImageError as e:
        problems.append(str(e))
    if cfg.pip_packages:
        target = layer_dir(cfg, layer_base)
        if not _layer_sentinel(target).exists():
            problems.append(f"pip_packages layer {target} is not provisioned; build it with: "
                            f"{build_command_hint(cfg)}")
    return PreflightReport(ok=not problems, backend=cfg.backend, runtime=runtime,
                           image_tag=tag, problems=problems)


def _build_sandbox_main(argv: list[str] | None = None) -> None:
    """Console-script entry (``tether-build-sandbox``): pre-build the sandbox image and layer.

    Accepts the same inputs a host puts in ``SandboxConfig`` so a deploy step builds exactly
    the image tag (and pip layer) the host will ask for. ``--check`` only runs the preflight.
    """
    defaults = SandboxConfig()
    parser = argparse.ArgumentParser(
        prog="tether-build-sandbox",
        description="Pre-build the tether sandbox image (and pip layer) for a SandboxConfig.")
    parser.add_argument("--preinstalled", nargs="*", default=list(defaults.preinstalled),
                        help="packages baked into the image (SandboxConfig.preinstalled)")
    parser.add_argument("--pip", nargs="*", default=[],
                        help="extra packages for the mounted layer (SandboxConfig.pip_packages)")
    parser.add_argument("--runtime", default=None, help="podman or docker (default: auto)")
    parser.add_argument("--timeout", type=float, default=defaults.build_timeout_s,
                        help="build timeout in seconds (SandboxConfig.build_timeout_s)")
    parser.add_argument("--check", action="store_true",
                        help="report readiness without building; exit 1 on problems")
    args = parser.parse_args(argv)

    config = replace(defaults, backend="container", preinstalled=tuple(args.preinstalled),
                     pip_packages=tuple(args.pip), container_runtime=args.runtime,
                     build_timeout_s=args.timeout, build_on_demand=True)
    if args.check:
        report = sandbox_preflight(config)
        if report.ok:
            print(f"sandbox ready: {report.image_tag} (via {report.runtime})")
            return
        for problem in report.problems:
            print(f"problem: {problem}")
        raise SystemExit(1)

    runtime = detect_runtime(config.container_runtime)
    tag = image_tag(config.preinstalled)
    ensure_image(runtime, tag, config)
    if config.pip_packages:
        ensure_layer(runtime, config)
    print(f"sandbox image ready: {tag} (via {runtime})")
