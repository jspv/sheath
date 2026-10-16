"""Sandbox backends for running agent scripts, root-confined.

``_OrchestratedSandbox`` owns the shared, backend-agnostic control-file orchestration; each
backend implements only ``_launch`` (how the child process is run). ``LocalSubprocessSandbox``
runs a scrubbed-env child with rlimits; the container backend lives in ``sandbox_container``.
See ``_OrchestratedSandbox`` for the (sequential, non-reentrant-per-root) run contract.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from . import workspace
from .config import SandboxConfig
from .handles import HandleIdReuseError, HandleStore
from .paths import safe_path, validate_segment
from .procio import run_bounded
from .workspace import ControlFileError

_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
_RUNNER = _RUNTIME_DIR / "_runner.py"
_SCRIPTS_DIR = ".scripts"
_MAX_REPORTED_IDS = 5   # distinct rejected ids named in an error message
_MAX_REPORTED_ID_LEN = 64
# Bounds on the child -> parent publish channel; requests are untrusted input.
_MAX_PUBLISH_CONTROL_BYTES = 1024 * 1024
_MAX_PUBLISH_REQUESTS = 64

# publisher(path, *, name, description, source) -> result record (see Session.publish)
Publisher = Callable[..., dict]


class NoSandboxIsolationWarning(UserWarning):
    """Raised when the local tier is selected: sandboxed code runs as the host user."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    result: Any | None
    error: str | None
    exit_code: int
    new_handles: list[str] = field(default_factory=list)
    killed_by: str | None = None
    published: list[dict] = field(default_factory=list)


class SandboxRuntimeUnavailable(RuntimeError):
    """Raised when the container backend is selected but no container runtime exists, or when
    isolation is required and the selected backend cannot provide it."""


class SandboxQuarantinedError(RuntimeError):
    """A sandbox run could not be confirmed stopped (an orphaned container may still be
    running in the workspace). Until the runtime confirms it is gone, no run starts and no
    parent-side file work touches the root."""


@dataclass
class ControlPlaneLimits:
    """Bounds on the parent<->child control channel. Defaults mirror TetherConfig."""
    max_emit_bytes: int = 1024 * 1024
    max_control_bytes: int = 8 * 1024 * 1024
    max_new_handles: int = 256


class SandboxExecutor(Protocol):
    """Swappable execution backend. Local now; container/remote later."""

    def run_script(self, path: str, args: list[str] | None = None) -> ExecResult: ...
    def run_code(self, code: str, args: list[str] | None = None) -> ExecResult: ...


@dataclass
class _LaunchResult:
    """What a backend's _launch returns: raw process outcome, pre-parse."""
    stdout: str
    stderr: str
    exit_code: int
    killed_by: str | None = None
    # True when the backend could not confirm the sandbox stopped (a timed-out container it
    # failed to remove). The child may still be writing, so nothing it produced is ingested.
    orphaned: bool = False


@dataclass
class _RunContext:
    """Everything a backend needs to launch one run. Paths are HOST paths; a backend
    translates them to its own namespace."""
    script_rel: str          # user script path relative to root
    argv: list[str]          # user args (already str-coerced)
    root: Path
    run_dir: Path            # per-run exchange directory (.tether/runs/<token>)
    config: SandboxConfig
    publish_enabled: bool = False


class _OrchestratedSandbox:
    """Shared run orchestration. Subclasses implement ``_launch``.

    Runs are serialized by ``self.lock`` (an ``RLock`` the Session also takes for its own file
    work in the root), so no sandboxed code is running while the parent validates or writes
    paths under the root. Each run gets a fresh exchange directory under ``.tether/runs/``;
    it is the only place the parent reads sandbox-written data from, and every such read is
    no-follow, regular-file-only and size-capped (``workspace.read_control_file``).
    """

    def __init__(self, root: Path | str, store: HandleStore,
                 config: SandboxConfig | None = None,
                 limits: ControlPlaneLimits | None = None) -> None:
        self.root = Path(root).resolve()
        self.store = store
        self.config = config or SandboxConfig()
        self.limits = limits or ControlPlaneLimits()
        self.publisher: Publisher | None = None   # set by Session when the host can receive files
        self.lock = threading.RLock()             # shared with the Session; see class docstring

    def check_quarantine(self) -> None:
        """Raise ``SandboxQuarantinedError`` while an earlier run may still be running.

        A no-op here: the local tier's child is always killed with its process group. The
        container backend overrides it. Call with ``self.lock`` held.
        """

    def run_code(self, code: str, args: list[str] | None = None) -> ExecResult:
        with self.lock:
            self.check_quarantine()
            workspace.verify_layout(self.root)   # .scripts/ must be a real directory
            scripts = self.root / _SCRIPTS_DIR
            # Collision-free across instances/processes; scripts persist as debuggable artifacts.
            fd, abspath = tempfile.mkstemp(prefix="inline_", suffix=".py", dir=scripts)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            os.chmod(abspath, 0o644)
            rel = str(Path(abspath).relative_to(self.root))
            return self.run_script(rel, args)

    def run_script(self, path: str, args: list[str] | None = None) -> ExecResult:
        with self.lock:
            self.check_quarantine()
            return self._run_script(path, args)

    def _run_script(self, path: str, args: list[str] | None) -> ExecResult:
        script = safe_path(self.root, path)  # raises PathEscapesRootError if outside
        argv = [str(a) for a in (args or [])]  # coerce so non-str args fail clearly, not opaquely

        workspace.verify_layout(self.root)
        run_dir = workspace.new_run_dir(self.root)
        try:
            registry = {hid: {"kind": h.kind, "path": h.path}
                        for hid, h in self.store.manifest_handles().items()}
            (run_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

            ctx = _RunContext(
                script_rel=str(script.relative_to(self.root)), argv=argv, root=self.root,
                run_dir=run_dir, config=self.config,
                publish_enabled=self.publisher is not None,
            )
            launched = self._launch(ctx)

            if launched.orphaned:
                return ExecResult(
                    stdout=launched.stdout, stderr=launched.stderr, result=None,
                    error=((launched.stderr.strip() + "\n") if launched.stderr.strip() else "")
                          + "tether: the sandbox could not be stopped; nothing it produced "
                            "was ingested",
                    exit_code=launched.exit_code, killed_by=launched.killed_by)

            # The local tier cannot stop a script replacing a protected directory; refuse to
            # ingest anything if one did. (In the container tier they are read-only mounts.)
            workspace.verify_layout(self.root)

            result, emit_error = self._read_emit(run_dir, launched.exit_code)

            # Ergonomic fallback: if the script neither emitted nor ended in an expression but
            # printed something, surface that so a model that just print()s an answer still gets one.
            if (result is None and emit_error is None and launched.exit_code == 0
                    and launched.stdout.strip()):
                result = launched.stdout.strip()

            new_handles, ingest_error = self._ingest_new_handles(run_dir)
            published, publish_error = self._process_publications(run_dir, launched.exit_code)
            base_error = (launched.stderr.strip() or None) if launched.exit_code != 0 else None
            error = "\n".join(p for p in (base_error, emit_error, ingest_error, publish_error)
                               if p) or None

            return ExecResult(stdout=launched.stdout, stderr=launched.stderr, result=result,
                              error=error, exit_code=launched.exit_code,
                              new_handles=new_handles, killed_by=launched.killed_by,
                              published=published)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)   # rmtree never follows a link

    def _read_emit(self, run_dir: Path, exit_code: int) -> tuple[Any | None, str | None]:
        """Parse the child's emit payload; return (result, error). Everything about the file
        is child-controlled, so every failure becomes an error string, never an exception."""
        if exit_code != 0:
            return None, None
        try:
            raw = workspace.read_control_file(run_dir, "emit.json", self.limits.max_emit_bytes)
        except ControlFileError as e:
            return None, f"tether: emit payload {e}; save a handle instead"
        if raw is None:
            return None, None
        try:
            return json.loads(raw.decode("utf-8", errors="replace")), None
        except (ValueError, RecursionError) as e:   # JSONDecodeError is a ValueError
            return None, f"tether: malformed emit payload: {type(e).__name__}: {e}"

    def _ingest_new_handles(self, run_dir: Path) -> tuple[list[str], str | None]:
        """Adopt handles the child saved; return (ids, error).

        Records are untrusted: the parent validates the id, picks the destination, moves the
        payload out of the run directory and derives all metadata itself
        (``HandleStore.adopt_file``). Tolerant by design -- a rejected record is counted and
        reported, not fatal. The control file is read only up to ``max_control_bytes`` and at
        most ``max_new_handles`` records are adopted.
        """
        ids: list[str] = []
        try:
            raw = workspace.read_control_file(run_dir, "new_handles.jsonl",
                                              self.limits.max_control_bytes)
        except ControlFileError as e:
            return ids, f"tether: control file {e}; no handles ingested"
        if raw is None:
            return ids, None
        reused: dict[str, None] = {}   # distinct rejected ids, insertion-ordered
        reused_count = 0
        rejected = 0
        cap_error: str | None = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if len(ids) >= self.limits.max_new_handles:
                cap_error = (f"tether: too many new handles (cap "
                             f"{self.limits.max_new_handles}); later records dropped")
                break
            try:
                rec = json.loads(line)
                name = validate_segment(rec["file"], what="handle payload name")
                handle = self.store.adopt_file(id=rec["id"], kind=rec["kind"],
                                               src=run_dir / name,
                                               source=rec.get("source", "run_python"))
                ids.append(handle.id)
            except HandleIdReuseError as e:
                reused_count += 1
                if len(reused) <= _MAX_REPORTED_IDS:  # stop tracking once past the display cap
                    reused[str(e.handle_id)[:_MAX_REPORTED_ID_LEN]] = None
            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError,
                    OSError):
                rejected += 1
        parts = []
        if reused_count:
            shown = list(reused)[:_MAX_REPORTED_IDS]
            more = ", ..." if len(reused) > _MAX_REPORTED_IDS else ""
            # Bounded by construction: the length cannot grow with the number of records.
            parts.append(f"tether: save rejected for {reused_count} record(s): handle id "
                         f"already exists and handles are immutable; ids: "
                         f"{', '.join(shown)}{more}")
        if rejected:
            parts.append(f"tether: {rejected} handle record(s) rejected (invalid id or kind, "
                         f"or the saved file was missing or unreadable)")
        if cap_error:
            parts.append(cap_error)
        return ids, "\n".join(parts) or None

    def _process_publications(self, run_dir: Path,
                              exit_code: int) -> tuple[list[dict], str | None]:
        """Hand the child's publish requests to the publisher; return (records, error).

        Runs only after the child has exited, and only on a clean exit, so published files are
        complete. Every request is untrusted: the publisher re-validates the path from scratch.
        """
        if self.publisher is None:
            return [], None
        try:
            raw = workspace.read_control_file(run_dir, "publish.jsonl",
                                              _MAX_PUBLISH_CONTROL_BYTES)
        except ControlFileError as e:
            return [], f"tether: publish control file {e}; publication requests ignored"
        if raw is None:
            return [], None
        lines = [ln for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if not lines:
            return [], None
        if exit_code != 0:
            return [], (f"tether: {len(lines)} publication request(s) ignored because the "
                        "script did not exit cleanly")
        notes: list[str] = []
        if len(lines) > _MAX_PUBLISH_REQUESTS:
            notes.append(f"tether: too many publication requests (cap {_MAX_PUBLISH_REQUESTS}); "
                         "later requests dropped")
        records: list[dict] = []
        for line in lines[:_MAX_PUBLISH_REQUESTS]:
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(req, dict):
                continue
            path, name, desc = req.get("path"), req.get("name"), req.get("description")
            if not isinstance(path, str):
                records.append({"error": "invalid publication request (path must be a string)"})
                continue
            try:
                records.append(self.publisher(
                    path, name=name if isinstance(name, str) else None,
                    description=desc if isinstance(desc, str) else None, source="run_python"))
            except Exception as e:  # noqa: BLE001 - one bad request must not lose the run
                records.append({"error": f"publication failed: {type(e).__name__}: {e}"})
        return records, "\n".join(notes) or None

    def _launch(self, ctx: _RunContext) -> _LaunchResult:
        raise NotImplementedError


class LocalSubprocessSandbox(_OrchestratedSandbox):
    """Runs the script in a scrubbed-env child process with rlimits + a wall-clock timeout.

    **This tier is not a security boundary** and provides no isolation: the code runs as the
    host user, sharing the kernel, the filesystem beyond the root, and the network. The
    rlimits and timeout bound resource use, not privilege. Use the container backend for a
    real boundary.
    """

    def _launch(self, ctx: _RunContext) -> _LaunchResult:
        script_abs = ctx.root / ctx.script_rel
        env = {
            "PATH": _minimal_path(),
            "HOME": str(ctx.root),
            "TMPDIR": str(ctx.root),
            "TETHER_ROOT": str(ctx.root),
            "TETHER_RUN_DIR": str(ctx.run_dir),
            "PYTHONPATH": str(_RUNTIME_DIR),
        }
        if ctx.publish_enabled:
            env["TETHER_PUBLISH_ENABLED"] = "1"
        run = run_bounded([sys.executable, str(_RUNNER), str(script_abs), *ctx.argv],
                          cwd=ctx.root, env=env, timeout=ctx.config.timeout_s,
                          cap=ctx.config.max_output_bytes, preexec_fn=self._limits())
        if run.timed_out:
            return _LaunchResult(run.stdout, run.stderr + "\ntether: killed (timeout)",
                                 -1, killed_by="timeout")
        return _LaunchResult(run.stdout, run.stderr, run.returncode)

    def _limits(self):
        cfg = self.config

        def set_limits() -> None:
            mem = cfg.max_memory_mb * 1024 * 1024
            fsize = cfg.max_file_size_mb * 1024 * 1024
            for res_id, limit in (
                (resource.RLIMIT_AS, mem),      # may be rejected on macOS — best-effort
                (resource.RLIMIT_FSIZE, fsize),
                (resource.RLIMIT_CORE, 0),
            ):
                try:
                    resource.setrlimit(res_id, (limit, limit))
                except (ValueError, OSError):
                    pass

        return set_limits


def _minimal_path() -> str:
    return "/usr/bin:/bin:/usr/local/bin"
