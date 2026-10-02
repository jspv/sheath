"""A Session bundles one run's root directory, handle store, and sandbox."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import stat
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import bundles as _bundles
from .config import TetherConfig, SandboxConfig
from .handles import Handle, HandleStore
from .paths import safe_filename, validate_segment
from .publish import OnPublish, PublishError, validate_publication
from .sandbox import (
    ControlPlaneLimits, LocalSubprocessSandbox, NoSandboxIsolationWarning,
    SandboxExecutor, SandboxRuntimeUnavailable,
)
from .status import StatusBus, StatusEvent, bind_bus

_DELIVER = "deliver"
_INPUTS = "inputs"
_COPY_CHUNK = 1024 * 1024


@dataclass
class Session:
    root: Path
    store: HandleStore
    sandbox: SandboxExecutor
    config: TetherConfig
    on_publish: OnPublish | None = None
    _mcp_connected: list[Any] = field(default_factory=list, init=False, repr=False)
    status_bus: StatusBus = field(default_factory=StatusBus, init=False, repr=False)
    _status_cm: Any = field(default=None, init=False, repr=False)
    # Serializes parent-side file work in the root with sandbox runs (shared with the sandbox).
    io_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def create(cls, config: TetherConfig, *, on_publish: OnPublish | None = None,
               bundles: tuple[str, ...] | None = None) -> "Session":
        """Open the workspace. Sandbox ``publish()`` is enabled only when ``on_publish`` is set
        and the ``deliver`` bundle is selected (``bundles=None`` means all bundles)."""
        root, pre_existing = _resolve_root(config)
        # The root has to exist before the HandleStore, which has to exist before the
        # sandbox -- so it cannot simply be created last. Instead, a root this call brought
        # into being is removed again if anything downstream fails. Without that, every
        # failed start on a machine with no container runtime left an empty
        # .tether/runs/run-XXXX behind, one per attempt. A root that already existed (a pinned
        # root_dir, a resumed thread) is never touched: it is not ours to delete.
        try:
            store = HandleStore(root)
            limits = ControlPlaneLimits(max_emit_bytes=config.max_emit_bytes,
                                        max_control_bytes=config.max_control_bytes,
                                        max_new_handles=config.max_new_handles)
            sandbox = _build_sandbox(root, store, config.sandbox, limits)
        except BaseException:
            if not pre_existing:
                shutil.rmtree(root, ignore_errors=True)
            raise
        session = cls(root=root, store=store, sandbox=sandbox, config=config,
                      on_publish=on_publish)
        if hasattr(sandbox, "lock"):
            session.io_lock = sandbox.lock        # one lock for sandbox runs + parent file work
        deliver = bundles is None or _DELIVER in _bundles.selected_bundles(bundles)
        if on_publish is not None and deliver and hasattr(sandbox, "publisher"):
            sandbox.publisher = session.publish   # enables publish() inside run_python
        return session

    @property
    def handles(self) -> dict[str, Any]:
        """Handle summaries produced during the run, by id."""
        return self.store.manifest()

    @property
    def artifacts(self) -> list[str]:
        """User-meaningful files under root, excluding handle storage and scratch."""
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(self.root)
            top = rel.parts[0]
            if top in ("handles", ".scripts", ".tether") or top.startswith("_"):
                continue
            out.append(rel.as_posix())
        return out

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> Callable[[], None]:
        """Register a status subscriber; returns a zero-arg unsubscribe handle."""
        return self.status_bus.subscribe(callback)

    async def aclose(self) -> None:
        """Close every connected MCP server, then honor the cleanup policy."""
        for tool in self._mcp_connected:
            try:
                await tool.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._mcp_connected.clear()
        if self.config.cleanup and self.root.exists():
            shutil.rmtree(self.root)

    def _unavailable_bundles(self) -> tuple[str, ...]:
        """Bundles that are selected-but-inert: ``deliver`` needs a host publish callback."""
        return () if self.on_publish is not None else (_DELIVER,)

    def tools(self, *bundles: str) -> list:
        """The built-in tool callables for the selected bundles (default: all)."""
        from .tools.registry import build_tools  # local import avoids circular dependency
        wanted = _bundles.tool_names_for(bundles, exclude=self._unavailable_bundles())
        return [t for t in build_tools(self) if t.__name__ in wanted]

    def tether_instructions(self, *bundles: str) -> str:
        """The operating-manual text (core + selected bundles)."""
        return _bundles.instructions_for(bundles, exclude=self._unavailable_bundles())

    def publish(self, path: str, *, name: str | None = None, description: str | None = None,
                source: str) -> dict:
        """Validate a workspace file and deliver it to the host's ``on_publish`` callback.

        Runs synchronously in the calling thread, so host contextvars set before
        ``agent.run`` are visible in the callback (tools run via ``asyncio.to_thread``, which
        copies the context; the callback must therefore be thread-safe). The callback gets a
        private snapshot outside the workspace (``PublishedFile.path``), copied from the
        validated bytes and deleted when the callback returns. Holds the sandbox lock, so no
        sandboxed code runs meanwhile. Never raises for a bad request or a failing callback:
        the result carries ``error`` so the agent loop can continue.
        """
        if self.on_publish is None:
            return {"error": "publishing is not enabled for this session", "path": path}
        with self.io_lock, tempfile.TemporaryDirectory(prefix="tether-publish-") as snap:
            try:
                pf = validate_publication(self.root, path, name=name, description=description,
                                          source=source, max_bytes=self.config.max_publish_bytes,
                                          snapshot_dir=Path(snap))
            except PublishError as e:
                return {"error": str(e), "path": path}
            record = {"name": pf.name, "rel_path": pf.rel_path, "size": pf.size,
                      "sha256": pf.sha256, "content_type": pf.content_type}
            try:
                host = self.on_publish(pf)
            except Exception as e:  # noqa: BLE001 - a host failure is reported, not fatal
                return {**record,
                        "error": f"host rejected the publication: {type(e).__name__}: {e}"}
        self.status_bus.emit(StatusEvent(tool="publish_file",
                                         message=f"published {pf.name} ({pf.size} bytes)"))
        return {**record, "host": host}

    @property
    def inputs(self) -> dict[str, Handle]:
        """User-provided inputs, keyed by host-supplied ``input_id``."""
        return self.store.inputs()

    def add_input(self, source: Path | bytes, *, input_id: str, name: str,
                  content_type: str | None = None, description: str | None = None) -> Handle:
        """Place a user upload at ``inputs/<input_id>/<name>`` and register it as a handle.

        The upload is untrusted and is **never parsed here**: it becomes a ``binary`` handle
        whose preview is built from raw bytes only. Idempotent by ``input_id`` (also across
        restarts, via the manifest). The file is made read-only (``0444``); the container tier
        additionally mounts ``inputs/`` read-only. Not safe to call while a turn is running —
        use ``Conversation.add_input`` / ``aadd_input``, which enforce that. Holds the sandbox
        lock, so no sandboxed code can swap paths under ``inputs/`` while it writes.
        """
        with self.io_lock:
            return self._add_input(source, input_id=input_id, name=name,
                                   content_type=content_type, description=description)

    def _add_input(self, source: Path | bytes, *, input_id: str, name: str,
                   content_type: str | None, description: str | None) -> Handle:
        validate_segment(input_id, what="input id")
        filename = safe_filename(name, fallback="upload.bin")
        size = _input_size(source)
        if size > self.config.max_input_bytes:
            raise ValueError(f"input too large ({size} bytes > {self.config.max_input_bytes})")

        # The manifest is child-writable in the container tier, so a recorded input is reused
        # only if its file is really there AND holds the host's bytes; otherwise the host's
        # bytes win. Either way the record is re-derived from the file below.
        existing = self.store.inputs().get(input_id)
        if (existing is not None and self._input_intact(existing, input_id)
                and _same_content(self.root / existing.path, source)):
            target = self.root / existing.path       # idempotent: no copy
            filename = target.name
        else:
            target_dir = self._input_dir(input_id)
            target = target_dir / filename
            _clear_squatter(target)
            fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".upload_")
            try:
                with os.fdopen(fd, "wb") as out:
                    _copy_bounded(source, out, self.config.max_input_bytes)
                os.chmod(tmp, 0o444)
                os.replace(tmp, target)   # replaces (never writes through) a planted symlink
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

        # The store derives bytes/preview/digest from `target` itself, by the same
        # describer manifest rehydration uses -- so a reopened store reproduces this record
        # instead of believing the manifest. Only what the host alone knows is passed in.
        return self.store.put_input(
            path=f"{_INPUTS}/{input_id}/{filename}",
            source=f"upload:{filename}", input_id=input_id,
            content_type=content_type or mimetypes.guess_type(filename)[0],
            description=description, replace=existing is not None)

    def _input_intact(self, handle: Handle, input_id: str) -> bool:
        """Whether a recorded input still points at its own regular file under inputs/<id>/.

        In the container tier the manifest is writable by sandboxed code, so the record is
        checked against the filesystem instead of being trusted.
        """
        parts = PurePosixPath(handle.path).parts
        if handle.kind != "binary" or len(parts) != 3 or parts[:2] != (_INPUTS, input_id):
            return False
        for rel in (_INPUTS, f"{_INPUTS}/{input_id}"):
            if (self.root / rel).is_symlink():
                return False
        try:
            return stat.S_ISREG(os.lstat(self.root / handle.path).st_mode)
        except OSError:
            return False

    def _input_dir(self, input_id: str) -> Path:
        """Create and return ``<root>/inputs/<input_id>`` as real, writable directories.

        Sandboxed code may have left a symlink, file, or read-only directory at either level;
        under the session lock it is removed (never followed) or repaired, so an upload can
        never be blocked or redirected.
        """
        path = self.root
        for part in (_INPUTS, input_id):
            path = path / part
            _ensure_real_dir(path)
        return path

    async def create_agent(
        self,
        client: Any,
        *,
        agent_instructions: str | None = None,
        tools: list | None = None,
        bundles: tuple[str, ...] = ("code", "files", "web"),
        name: str = "data-integrator",
        **maf_kwargs: Any,
    ):
        """Build a MAF agent over the selected bundles plus developer tools/MCP.

        Plain callables are spill-wrapped; MCP servers are connected and their tools get
        the spill parser (Task 5). Operational instructions ride in ``tether_instructions``.
        """
        from agent_framework import create_harness_agent  # local: heavy dep, imported at call time

        from .spill import looks_like_mcp, spill_tool  # local: spill imports Session -> circular at module level

        builtin = self.tools(*bundles)
        external: list = []
        for tool in tools or []:
            if looks_like_mcp(tool):
                external.extend(await self._attach_mcp(tool))
            else:
                external.append(spill_tool(self, tool))

        maf_kwargs.setdefault("max_context_window_tokens", self.config.max_context_window_tokens)
        maf_kwargs.setdefault("max_output_tokens", self.config.max_output_tokens)
        return create_harness_agent(
            client,
            name=name,
            harness_instructions=self.tether_instructions(*bundles),
            agent_instructions=agent_instructions,
            tools=builtin + external,
            disable_todo=True,
            disable_mode=True,
            disable_memory=True,
            disable_web_search=True,
            **maf_kwargs,
        )

    async def _attach_mcp(self, tool: Any) -> list:
        """Connect an MCP server, attach the spill parser, capture its status, own its lifecycle."""
        from .mcp_status import inject_progress_tokens, install_status_wrappers
        from .spill import make_spill_parser

        server = getattr(tool, "name", None) or repr(tool)
        # Install before connect so the wrappers reach the underlying MCP ClientSession.
        token_map = install_status_wrappers(self.status_bus, tool, server)
        try:
            await tool.connect()
        except Exception as e:  # noqa: BLE001 - add context naming the server, then re-raise
            raise RuntimeError(f"failed to connect MCP server {tool!r}: {e}") from e
        # Register before the parser loop so aclose() still closes this server if the loop raises.
        self._mcp_connected.append(tool)
        inject_progress_tokens(tool, server, token_map)   # after connect: tools are now loaded
        functions = list(tool.functions)
        for ft in functions:
            ft.result_parser = make_spill_parser(self, ft.name)
        return functions

    async def __aenter__(self) -> "Session":
        self._status_cm = bind_bus(self.status_bus)
        self._status_cm.__enter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.aclose()
        finally:
            if self._status_cm is not None:
                self._status_cm.__exit__(None, None, None)
                self._status_cm = None

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


def _resolve_root(config: TetherConfig) -> tuple[Path, bool]:
    """Return ``(root, pre_existing)``, creating the root.

    A pinned ``root_dir`` is used verbatim. Otherwise a fresh directory is created under
    ``./.tether/runs/`` with ``mkdtemp``, which is atomic: two threads or processes can
    never be handed the same root. That base is deliberately not ``./.tether/sessions/``,
    where conversation roots are named by a caller-supplied id.
    """
    if config.root_dir is not None:
        root = Path(config.root_dir).resolve()
        pre_existing = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        return root, pre_existing
    base = Path.cwd() / ".tether" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base)).resolve(), False


def _input_size(source: Path | bytes) -> int:
    if isinstance(source, (bytes, bytearray)):
        return len(source)
    st = os.stat(source)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"input source is not a regular file: {source}")
    return st.st_size


def _ensure_real_dir(path: Path) -> None:
    """Make ``path`` a real directory we can write: replace anything else, never follow it."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        path.mkdir()
        return
    if not stat.S_ISDIR(st.st_mode):
        path.unlink()          # symlink, file, FIFO, ...: removed, not followed
        path.mkdir()
        return
    os.chmod(path, 0o755)      # lstat just proved it is a real directory, not a link


def _clear_squatter(target: Path) -> None:
    """Remove a directory squatting on an input's filename (``os.replace`` cannot)."""
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        _make_tree_writable(target)
        shutil.rmtree(target)      # rmtree does not follow symlinks


def _make_tree_writable(top: Path) -> None:
    """Make every real directory under ``top`` listable and writable so rmtree can empty it.

    Walks top-down without following links (``os.walk`` default), chmods each directory
    before descending into it, and touches nothing that ``lstat`` doesn't show to be a real
    directory.
    """
    os.chmod(top, 0o700)       # the caller's lstat proved it is a real directory
    for dirpath, dirnames, _ in os.walk(top):
        for d in dirnames:
            child = os.path.join(dirpath, d)
            if stat.S_ISDIR(os.lstat(child).st_mode):
                os.chmod(child, 0o700)


def _sha256_file(path: Path | str) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.digest()


def _same_content(path: Path, source: Path | bytes) -> bool:
    """Whether the file at ``path`` holds exactly the host's ``source`` bytes."""
    want = (hashlib.sha256(source).digest() if isinstance(source, (bytes, bytearray))
            else _sha256_file(source))
    return _sha256_file(path) == want


def _copy_bounded(source: Path | bytes, out, limit: int) -> None:
    """Copy ``source`` into ``out``; raise if more than ``limit`` bytes arrive."""
    if isinstance(source, (bytes, bytearray)):
        out.write(source)      # length already checked against the limit
        return
    copied = 0
    with open(source, "rb") as src:
        while chunk := src.read(_COPY_CHUNK):
            copied += len(chunk)
            if copied > limit:
                raise ValueError(f"input too large (more than {limit} bytes while copying)")
            out.write(chunk)


def _check_isolation(sandbox_config: SandboxConfig) -> None:
    """Refuse a backend that cannot provide the isolation the config requires."""
    if sandbox_config.require_isolation and sandbox_config.backend != "container":
        raise SandboxRuntimeUnavailable(
            f"SandboxConfig.require_isolation=True but backend={sandbox_config.backend!r}: the "
            'local tier is not a security boundary. Set backend="container" (podman or docker), '
            "or drop require_isolation for development/trusted use only."
        )


def _build_sandbox(root: Path, store: HandleStore, sandbox_config: SandboxConfig,
                   limits: ControlPlaneLimits | None = None) -> SandboxExecutor:
    """Pick the sandbox backend from config (default: container). The one place the backend
    is decided; nothing else validates or selects it.

    Four guards, in order: (1) the isolation requirement -- ``require_isolation=True``
    refuses anything but the container tier, however it was selected (an explicit
    ``backend="local"`` or ``TETHER_SANDBOX_BACKEND=local``); (2) the backend name, which
    fails closed on anything but ``local``/``container``; (3) runtime *liveness*, not mere
    presence; (4) construction. A missing container runtime is a hard error, never a
    fallback: silently dropping to the local tier would hand back a no-isolation sandbox
    while the caller believes the code is contained. The opt-out has to be explicit.

    Guard (3) answers "is the runtime usable". Whether the *image* is ready is a separate
    question, answered on first use (``build_on_demand``) or ahead of time by
    ``sandbox_preflight``; neither guard substitutes for the other.
    """
    _check_isolation(sandbox_config)
    if sandbox_config.backend not in ("local", "container"):
        raise ValueError(
            f"unknown sandbox backend {sandbox_config.backend!r}; expected 'container' "
            f"(real isolation, the default) or 'local' (NO isolation). Note "
            f"TETHER_SANDBOX_BACKEND selects the tier, not the container runtime -- "
            f"use SandboxConfig.container_runtime for that."
        )
    if sandbox_config.backend == "container":
        from . import container_runtime
        from .sandbox_container import ContainerSandbox  # local import: optional backend
        try:
            # Probe LIVENESS, not mere presence. `detect_runtime` only checks shutil.which,
            # so a machine with podman installed but its VM not started passes detection and
            # then fails much later with an opaque "failed to build sandbox image" error.
            # That is the most common failure mode -- especially on macOS, where the runtime
            # lives in a Linux VM -- and it is exactly the case this message exists to serve.
            container_runtime.require_usable_runtime(sandbox_config.container_runtime)
        except RuntimeError as e:
            # One coherent message: the probe states the specific fault (not on PATH, or
            # installed but not responding, with its own `machine start` hint), and this
            # adds only what the probe cannot know -- why container is the default, and the
            # exact opt-out. Neither half repeats the other.
            raise SandboxRuntimeUnavailable(
                f"{e}\n\nrun_python executes model-authored code, which is why the "
                f"container backend is the default. Install and start a runtime, or set "
                f'TetherConfig.sandbox.backend = "local" (or TETHER_SANDBOX_BACKEND=local) '
                f"to run it with no isolation."
            ) from e
        # Deliberately outside the try: only the probe's RuntimeErrors mean "no usable
        # runtime". A RuntimeError from the constructor is a different bug and must not be
        # relabelled as a missing runtime, sending the user to install something they have.
        return ContainerSandbox(root=root, store=store, config=sandbox_config, limits=limits)
    warnings.warn(
        "sandbox backend 'local' provides NO isolation: run_python executes "
        "model-authored code as the host user. Use the 'container' backend for a real "
        "boundary.",
        NoSandboxIsolationWarning,
        stacklevel=2,
    )
    return LocalSubprocessSandbox(root=root, store=store, config=sandbox_config, limits=limits)
