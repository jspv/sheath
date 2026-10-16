"""ContainerSandbox: run agent code in a hardened OCI container (podman/docker).

Reuses the shared orchestration in ``_OrchestratedSandbox`` (run directory, ExecResult parsing)
and only differs in ``_launch``. Mounts:

- the session root at /workspace (read-write);
- the parent-owned trees ``handles/``, ``inputs/``, ``.scripts/`` and ``.tether/`` overlaid on it
  read-only, so sandboxed code can neither write into them nor replace them;
- this run's exchange directory (``.tether/runs/<token>``) at /tether-run (read-write) -- the
  only place the parent reads sandbox-written data from;
- the tether runtime/ at /runtime (read-only).

It runs ``python /runtime/_runner.py <script-rel> <args>`` with network off (by default), a
read-only root filesystem, dropped capabilities, and memory/cpu/pid limits.
"""

from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path

from . import workspace
from .config import SandboxConfig
from .procio import run_bounded
from .sandbox import (
    _RUNTIME_DIR,
    ControlPlaneLimits,
    _LaunchResult,
    _OrchestratedSandbox,
    _RunContext,
    SandboxQuarantinedError,
)

_PIDS_LIMIT = 256
_RM_TIMEOUT_S = 30.0
_RECHECK_TIMEOUT_S = 10.0    # per runtime call when re-checking a quarantined container


def _says_no_such_container(output: str) -> bool:
    """True only for the runtime's own missing-container message (podman: ``no such
    container``; docker: ``No such container:``). Never the bare "no such", which also
    matches a socket error ("connect: no such file or directory") from an unreachable
    runtime that says nothing about whether the container is still running."""
    return "no such container" in output.lower()


class ContainerSandbox(_OrchestratedSandbox):
    def __init__(self, root: Path | str, store, config: SandboxConfig | None = None,
                 runtime: str | None = None,
                 limits: ControlPlaneLimits | None = None) -> None:
        super().__init__(root, store, config, limits)
        if runtime is not None:
            self._runtime = runtime
        else:
            from .container_runtime import detect_runtime
            self._runtime = detect_runtime(self.config.container_runtime)
        # Exact names of containers from timed-out runs that the runtime has not confirmed
        # gone. While any is listed the session is quarantined (see check_quarantine).
        self._orphans: list[str] = []

    def _build_run_argv(self, ctx: _RunContext, tag: str, layer: Path | None,
                        name: str | None = None) -> list[str]:
        cfg = self.config
        argv = [self._runtime, "run", "--rm"]
        if name is not None:
            argv += ["--name", name]
        argv += [
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(_PIDS_LIMIT),
            "--memory", f"{cfg.max_memory_mb}m", "--cpus", str(cfg.max_cpus),
        ]
        # Rootless podman maps the host user to container-root by default, so a bind-mounted
        # session root (owned by the host uid, 0600 scripts) is unreadable/unwritable to the
        # hardening --user below. keep-id maps the host uid straight through, so ownership of the
        # mounted workspace lines up with the --user we run as. (podman-only flag; docker rootless
        # uses a different model and rejects it.)
        if self._runtime == "podman":
            argv += ["--userns=keep-id"]
        argv += [
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ctx.root}:/workspace:rw",
            "-v", f"{_RUNTIME_DIR}:/runtime:ro",
            "-w", "/workspace",
        ]
        # Parent-owned trees are overlaid read-only on the rw workspace. Sandboxed code can
        # then neither write inside them nor replace them (a mount point cannot be renamed
        # or removed from inside the container). The orchestrator has already verified that
        # each is a real directory, so none of these mounts a link's target.
        for name in workspace.PROTECTED_DIRS:
            argv += ["-v", f"{ctx.root / name}:/workspace/{name}:ro"]
        # The one writable channel back to the parent: this run's exchange directory.
        argv += ["-v", f"{ctx.run_dir}:/tether-run:rw"]
        if not cfg.network:
            argv += ["--network", "none"]
        pythonpath = "/runtime"
        if layer is not None:
            argv += ["-v", f"{layer}:/pkgs:ro"]
            pythonpath = "/runtime:/pkgs"
        env = {
            "TETHER_ROOT": "/workspace",
            "TETHER_RUN_DIR": "/tether-run",
            "PYTHONPATH": pythonpath,
            "HOME": "/workspace",
            "TMPDIR": "/tmp",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }
        if ctx.publish_enabled:
            env["TETHER_PUBLISH_ENABLED"] = "1"
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        argv += [tag, "python", "/runtime/_runner.py", ctx.script_rel, *ctx.argv]
        return argv

    def _launch(self, ctx: _RunContext) -> _LaunchResult:
        from .container_runtime import ensure_image, ensure_layer, image_tag

        tag = image_tag(self.config.preinstalled)
        ensure_image(self._runtime, tag, self.config)
        layer = ensure_layer(self._runtime, self.config) if self.config.pip_packages else None
        name = f"tether-{secrets.token_hex(8)}"
        argv = self._build_run_argv(ctx, tag, layer, name=name)
        run = run_bounded(argv, timeout=ctx.config.timeout_s,
                          cap=ctx.config.max_output_bytes)
        if run.timed_out:
            # Killing the runtime CLI does not stop the container: remove it by name. If that
            # fails the container may still be running in the bind-mounted root, so the run
            # is reported as orphaned and the orchestrator ingests nothing from it.
            note = self._remove_container(name)
            if note:
                self._orphans.append(name)
            return _LaunchResult(run.stdout, run.stderr + "\ntether: killed (timeout)" + note,
                                 -1, killed_by="timeout", orphaned=bool(note))
        # 137 == 128 + SIGKILL: the runtime killed the container, typically the memory cap.
        killed_by = "killed" if run.returncode == 137 else None
        return _LaunchResult(run.stdout, run.stderr, run.returncode, killed_by=killed_by)

    def check_quarantine(self) -> None:
        """Refuse while a container from a timed-out run may still be running.

        Each recorded container is re-removed by its exact name (never anything else); one
        the runtime confirms gone is dropped from the list. If any remains, raise
        ``SandboxQuarantinedError``: that container may still be writing in the
        bind-mounted root, so neither a new run nor parent-side file work may proceed.
        """
        with self.lock:
            if not self._orphans:
                return
            still: list[str] = []
            details: list[str] = []
            for name in self._orphans:
                gone, detail = self._confirm_gone(name, _RECHECK_TIMEOUT_S)
                if not gone:
                    still.append(name)
                    details.append(f"{name}: {detail}" if detail else name)
            self._orphans = still
            if still:
                raise SandboxQuarantinedError(
                    "sandbox quarantined: a timed-out run's container could not be confirmed "
                    f"stopped and may still be writing in the workspace ({'; '.join(details)}). "
                    "Nothing will run in or touch the workspace until the container runtime "
                    f"removes it (e.g. `{self._runtime} rm -f {still[0]}`); every call retries.")

    def _confirm_gone(self, name: str, timeout: float) -> tuple[bool, str]:
        """Force-remove container ``name``; return (confirmed gone, failure detail).

        Gone means ``rm -f`` exited 0, or the runtime reports there is no such *container*
        (``rm -f`` of an already-removed container fails on some runtimes). Any other failure
        -- including an unreachable runtime, whose socket error reads "no such file or
        directory" -- is not gone: the container may still be running.
        """
        detail = ""
        try:
            proc = subprocess.run([self._runtime, "rm", "-f", name], capture_output=True,
                                  text=True, timeout=timeout)
            if proc.returncode == 0:
                return True, ""
            detail = proc.stderr.strip()
            if _says_no_such_container(proc.stderr + proc.stdout):
                return True, ""
        except (subprocess.TimeoutExpired, OSError) as e:
            detail = str(e)
        try:
            proc = subprocess.run([self._runtime, "container", "inspect", name],
                                  capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return False, detail
        if proc.returncode != 0 and _says_no_such_container(proc.stderr + proc.stdout):
            return True, ""
        return False, detail

    def _remove_container(self, name: str) -> str:
        """Force-remove a container by name; returns a warning suffix if that failed."""
        gone, detail = self._confirm_gone(name, _RM_TIMEOUT_S)
        if gone:
            return ""
        return f"\ntether: WARNING: could not remove timed-out container {name}: {detail}"
