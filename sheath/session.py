"""A Session bundles one run's root directory, handle store, and sandbox."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import mimetypes
import os
import shutil
import stat
import tempfile
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

from . import bundles as _bundles
from . import input_trees as _input_trees
from .config import SheathConfig, SandboxConfig
from .handles import Handle, HandleStore
from .paths import safe_filename, safe_path, validate_segment
from .publish import OnPublish, PublishError, validate_publication
from .sandbox import (
    ControlPlaneLimits, LocalSubprocessSandbox, NoSandboxIsolationWarning,
    SandboxExecutor, SandboxQuarantinedError, SandboxRuntimeUnavailable,
)
from .status import StatusBus, StatusEvent, bind_bus

if TYPE_CHECKING:
    from .delegation import Delegation
    from .subagents import SpawnPolicy, SubagentSpec

_DELIVER = "deliver"
_INPUTS = "inputs"
_COPY_CHUNK = 1024 * 1024
# create_harness_agent options that add a capability path around the sandbox, safe_path or egress.
_UNSANDBOXED_KWARGS = frozenset({"shell_executor", "file_access_store", "background_agents"})

_log = logging.getLogger(__name__)


def _remove_root_locked(root: Path, lock: Any,
                        check_quarantine: Callable[[], None] | None = None) -> None:
    """Remove ``root`` while holding ``lock`` (the session's ``io_lock``).

    Sync on purpose, and run with ``asyncio.to_thread``: a sync file tool still in flight on a
    worker thread (its task already cancelled) holds the lock, and the directory must not go
    until it has finished. Taking the lock on the event loop would stall the loop instead.

    Best effort: sandboxed code can ``chmod 000`` a directory in the root, so every real
    directory under it is first made listable and writable (everything in the root is owned by
    the host user, so the reset can always succeed). That walk checks each entry with ``lstat``
    before ``chmod``, which is safe only while nothing else writes in the root; the lock and
    the cancelled sub-agent tasks ensure that, except under a quarantined sandbox (a
    timed-out container that may still be running). ``check_quarantine`` (the sandbox's,
    called with the lock held) detects that case, and the permission reset is then skipped;
    a check that fails any other way is treated the same (logged, never raised).
    An entry that still cannot be removed does not stop the walk; everything else goes, and
    a root left behind is logged as a warning rather than raised.
    """
    failures: list[str] = []

    def _record(func: Any, path: str, exc: BaseException) -> None:
        failures.append(f"{path!r}: {type(exc).__name__}")  # repr: names are untrusted

    with lock:
        if not root.exists():
            return
        quarantined = False
        if check_quarantine is not None:
            try:
                check_quarantine()
            except SandboxQuarantinedError as e:
                quarantined = True
                _log.warning("removing %s: the sandbox is quarantined (a container may still "
                             "be writing); permissions not reset, so directories sandboxed code "
                             "made unreadable may be left behind. %s Once the container is "
                             "removed, delete %s to finish cleanup", root, e, root)
            except Exception as e:  # noqa: BLE001 - fail safe: a close path must not raise
                quarantined = True
                _log.warning("removing %s: the quarantine check failed (%s); treating the "
                             "sandbox as quarantined, permissions not reset", root,
                             type(e).__name__)
        if not quarantined:
            _make_tree_writable(root, best_effort=True)
        shutil.rmtree(root, onexc=_record)
    if root.exists():
        _log.warning("could not fully remove %s (%d entr%s left: %s)", root, len(failures),
                     "y" if len(failures) == 1 else "ies", "; ".join(failures[:5]))


@dataclass
class Session:
    root: Path
    store: HandleStore
    sandbox: SandboxExecutor
    config: SheathConfig
    on_publish: OnPublish | None = None
    _mcp_connected: list[Any] = field(default_factory=list, init=False, repr=False)
    status_bus: StatusBus = field(default_factory=StatusBus, init=False, repr=False)
    _status_cm: Any = field(default=None, init=False, repr=False)
    # Serializes parent-side file work in the root with sandbox runs (shared with the sandbox).
    io_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    # Sub-agents create_agent built over this session, by name.
    subagent_names: tuple[str, ...] = field(default=(), init=False)
    # Sub-agent runs registered by ConfinedSubagent, queued or running.
    _subagent_tasks: set[asyncio.Task] = field(default_factory=set, init=False, repr=False)
    # Shared-workspace tracking (sheath.tools.files), guarded by io_lock: root-relative path ->
    # (last writer, sha256); and (agent, path) -> sha256 that agent last read or wrote.
    _file_writers: dict = field(default_factory=dict, init=False, repr=False)
    _file_seen: dict = field(default_factory=dict, init=False, repr=False)
    # The agent tree and its tasks (sheath.delegation), created on first use.
    _delegation: Any = field(default=None, init=False, repr=False)
    # The subagent_max_concurrent semaphore, created lazily inside the running loop.
    _subagent_slots: Any = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, config: SheathConfig, *, on_publish: OnPublish | None = None,
               bundles: tuple[str, ...] | None = None) -> "Session":
        """Open the workspace. Sandbox ``publish()`` is enabled only when ``on_publish`` is set
        and the ``deliver`` bundle is selected (``bundles=None`` means all bundles)."""
        root, pre_existing = _resolve_root(config)
        # The root has to exist before the HandleStore, which has to exist before the
        # sandbox -- so it cannot simply be created last. Instead, a root this call brought
        # into being is removed again if anything downstream fails. Without that, every
        # failed start on a machine with no container runtime left an empty
        # .sheath/runs/run-XXXX behind, one per attempt. A root that already existed (a pinned
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
        """User-meaningful files under root, excluding handle storage, inputs and scratch."""
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(self.root)
            top = rel.parts[0]
            if top in ("handles", "inputs", ".scripts", ".sheath") or top.startswith("_"):
                continue
            out.append(rel.as_posix())
        return out

    @contextmanager
    def file_io(self) -> Iterator[None]:
        """Hold ``io_lock`` for parent-side file work in the root, after the sandbox's
        quarantine check: while a timed-out container may still be running in the root,
        this raises ``SandboxQuarantinedError`` instead of letting the work start."""
        with self.io_lock:
            check = getattr(self.sandbox, "check_quarantine", None)
            if check is not None:
                check()
            yield

    @property
    def subagents_running(self) -> int:
        """Sub-agent runs registered with this session, queued or running."""
        return len(self._subagent_tasks)

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> Callable[[], None]:
        """Register a status subscriber; returns a zero-arg unsubscribe handle."""
        return self.status_bus.subscribe(callback)

    @property
    def delegation(self) -> "Delegation":
        """This session's agent tree and tasks (``sheath.delegation.Delegation``)."""
        if self._delegation is None:
            from .delegation import Delegation  # local: agent_framework is a heavy import
            self._delegation = Delegation(self)
        return self._delegation

    async def cancel_subagents(self) -> None:
        """Cancel every running sub-agent task and wait for it, bounded by
        ``sheath.delegation.CANCEL_WAIT_S``. A task still pending after that is logged and
        left. See ``Delegation.cancel_all``."""
        await self.delegation.cancel_all()

    async def release_subagents(self, agent_session: Any) -> None:
        """Cancel this conversation's running sub-agent tasks and refuse new ones; for a
        closing conversation. ``agent_session`` is the conversation's ``AgentSession``; task
        state lives in this session, so every task goes. Failures are logged, not raised."""
        try:
            self.delegation.closed = True
            await self.delegation.cancel_all()
        except Exception:  # noqa: BLE001 - best-effort teardown
            _log.warning("releasing sub-agent tasks failed", exc_info=True)

    async def aclose(self) -> None:
        """Cancel sub-agent tasks, close every connected MCP server, then honor the cleanup
        policy. In that order: no sub-agent outlives its MCP servers or its root.

        Removing the root is best effort, as for ``Conversation.aclose``: an entry that cannot
        be removed is skipped, and a root left behind is logged as a warning, not raised."""
        try:
            self.delegation.closed = True      # no new task starts once it closes
            await self.cancel_subagents()
        finally:
            for tool in self._mcp_connected:
                try:
                    await tool.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
            self._mcp_connected.clear()
            if self.config.cleanup:
                await asyncio.to_thread(_remove_root_locked, self.root, self.io_lock,
                                        getattr(self.sandbox, "check_quarantine", None))

    def _unavailable_bundles(self) -> tuple[str, ...]:
        """Bundles that are selected-but-inert: ``deliver`` needs a host publish callback."""
        return () if self.on_publish is not None else (_DELIVER,)

    def tools(self, *bundles: str) -> list:
        """The built-in tool callables for the selected bundles (default: all)."""
        from .tools.registry import build_tools  # local import avoids circular dependency
        wanted = _bundles.tool_names_for(bundles, exclude=self._unavailable_bundles())
        return [t for t in build_tools(self) if t.__name__ in wanted]

    def sheath_instructions(self, *bundles: str) -> str:
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
        sandboxed code runs meanwhile, and refuses while the sandbox is quarantined. Never raises for a bad request or a failing callback:
        the result carries ``error`` so the agent loop can continue.
        """
        if self.on_publish is None:
            return {"error": "publishing is not enabled for this session", "path": path}
        try:
            with self.file_io():
                out = self._publish(path, name=name, description=description, source=source)
        except SandboxQuarantinedError as e:
            return {"error": str(e), "path": path}
        if "error" not in out:
            self.status_bus.emit(StatusEvent(tool="publish_file",
                                             message=f"published {out['name']} "
                                                     f"({out['size']} bytes)"))
        return out

    def _publish(self, path: str, *, name: str | None, description: str | None,
                 source: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="sheath-publish-") as snap:
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
        lock, so no sandboxed code can swap paths under ``inputs/`` while it writes; raises
        ``SandboxQuarantinedError`` while the sandbox is quarantined.
        """
        with self.file_io():
            return self._add_input(source, input_id=input_id, name=name,
                                   content_type=content_type, description=description)

    def _add_input(self, source: Path | bytes, *, input_id: str, name: str,
                   content_type: str | None, description: str | None) -> Handle:
        validate_segment(input_id, what="input id")
        if input_id in _input_trees.load(self.root):
            raise ValueError(f"input id {input_id!r} is already used by an input tree")
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

    @property
    def input_trees(self) -> dict[str, "_input_trees.InputTree"]:
        """Directory trees staged with ``add_input_tree``, keyed by ``input_id``."""
        return _input_trees.load(self.root)

    def add_input_tree(self, source: Path | str, *, input_id: str,
                       description: str | None = None,
                       ignore: Callable[[str], bool] | None = None,
                       replace: bool = False) -> "_input_trees.InputTree":
        """Copy the directory ``source`` read-only to ``inputs/<input_id>/``.

        Only regular files are copied; symlinks, FIFOs, sockets and devices are skipped and
        never followed. ``ignore(rel_path)`` sees each directory's and file's POSIX path
        relative to ``source``; ``True`` skips it, and a skipped directory is pruned whole.
        Staged files are ``0444`` and directories ``0555``; ``write_file`` and ``edit_file``
        refuse them, and the container tier mounts ``inputs/`` read-only. The tree is not a
        handle: no ``load(id)``. It is excluded from ``artifacts``, and the agent is told
        where it is and that it is data. Bounded by ``max_input_tree_files`` and
        ``max_input_tree_bytes``; exceeding either raises ``ValueError`` and leaves nothing
        behind. A repeat call with the same id returns the existing record without copying,
        unless ``replace``. Holds the sandbox lock, so it raises ``SandboxQuarantinedError``
        while the sandbox is quarantined. Not safe during a turn: use
        ``Conversation.add_input_tree`` / ``aadd_input_tree``.
        """
        with self.file_io():
            return self._add_input_tree(Path(source), input_id=input_id,
                                        description=description, ignore=ignore,
                                        replace=replace)

    def _add_input_tree(self, source: Path, *, input_id: str, description: str | None,
                        ignore: Callable[[str], bool] | None,
                        replace: bool) -> "_input_trees.InputTree":
        validate_segment(input_id, what="input id")
        if not source.is_dir():
            raise ValueError(f"input tree source must be a directory: {str(source)!r}")
        source = source.resolve()
        if self.root.resolve().is_relative_to(source):
            raise ValueError(f"input tree source {str(source)!r} contains the session root")
        if input_id in self.store.inputs():
            raise ValueError(f"input id {input_id!r} is already used by a single-file input")
        trees = _input_trees.load(self.root)
        if input_id in trees and not replace:
            return trees[input_id]
        _ensure_real_dir(self.root / _INPUTS)
        files, total = _input_trees.stage(
            self.root, source, input_id=input_id, ignore=ignore,
            max_files=self.config.max_input_tree_files,
            max_bytes=self.config.max_input_tree_bytes)
        tree = _input_trees.InputTree(input_id=input_id, path=f"{_INPUTS}/{input_id}",
                                      files=files, bytes=total, source=str(source),
                                      description=description)
        trees[input_id] = tree
        _input_trees.save(self.root, trees)
        return tree

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
        subagents: Mapping[str, "SubagentSpec"] | None = None,
        client_factory: Callable[[str], Any] | None = None,
        spawn: "SpawnPolicy | None" = None,
        _mcp_agent: str | None = None,
        _harness_extra: str | None = None,
        _prebuilt_tools: list | None = None,
        _node: Any = None,
        _message_node: Any = None,
        **maf_kwargs: Any,
    ):
        """Build a MAF agent over the selected bundles plus developer tools/MCP.

        Plain callables are spill-wrapped; MCP servers are connected and their tools get
        the spill parser (Task 5). Operational instructions ride in ``sheath_instructions``.

        ``subagents`` names sub-agents the model may delegate to. Each is built here, over this
        same session, so it can do nothing this agent could not; each gets a client of its own,
        ``spec.client`` or ``client_factory(spec.model or config.model)``. While a delegated
        task is still running, the agent is re-run rather than allowed to finish.

        ``spawn`` (a ``SpawnPolicy``) gives the agent ``spawn_subagent``, to create sub-agents
        of its own at runtime within the policy's bounds; each gets a client of its own from
        ``client_factory(model)``, so a policy needs a factory. ``None``: no such tool.

        ``_prebuilt_tools`` are tool objects another agent already built (spill-wrapped, or
        connected MCP functions), handed down as they are.

        ``_node`` is the ``AgentNode`` of a sub-agent being built that may spawn its own: it
        gets the delegation tools and the parent-wait loop bound to that node, exactly as a
        delegating top agent does, so it waits for its own sub-agents before reporting.

        ``_message_node`` is the ``AgentNode`` of any sub-agent being built: it gets
        ``send_message`` (and ``set_message_permissions`` when it has or may create sub-agents)
        whenever the session has a spawn policy. A top agent with a tree gets them likewise.
        """
        from agent_framework import create_harness_agent  # local: heavy dep, imported at call time

        from .spill import looks_like_mcp, spill_tool  # local: spill imports Session -> circular at module level

        refused = sorted(_UNSANDBOXED_KWARGS & maf_kwargs.keys())
        if refused:
            raise TypeError(
                f"{', '.join(refused)}: refused, would give the model shell, file or agent access "
                "outside the sandbox, safe_path and egress checks")
        policy_known = spawn is not None or (
            self._delegation is not None and self._delegation.policy is not None)
        granted = [n for n, sp in (subagents or {}).items() if getattr(sp, "message_to", ())]
        if granted and not policy_known:
            raise ValueError(
                f"sub-agent(s) {granted} set message_to, but messaging needs a SpawnPolicy: "
                "pass spawn=SpawnPolicy(...) (and a client_factory) to create_agent, or drop "
                "message_to")
        if spawn is not None:
            from .subagents import validate_spawn_policy
            validate_spawn_policy(spawn, client_factory)
            self._set_spawn_policy(spawn, client_factory, check_only=True)   # before building
        wait_timeout = maf_kwargs.pop("background_agents_wait_timeout_seconds", None)
        providers: list = []
        top = None
        if subagents or spawn is not None:
            top = self.delegation.top(name)
            top.client = client
            top.bundles = _bundles.selected_bundles(bundles)
            if subagents:
                await self._build_subagents(client, subagents, top=top, bundles=bundles,
                                            client_factory=client_factory, spawn=spawn)
            if spawn is not None:
                self._set_spawn_policy(spawn, client_factory)
                top.can_spawn = True
            providers.append(self._delegation_provider(top, wait_timeout))
            _add_parent_loop(maf_kwargs, top)
        elif _node is not None:
            providers.append(self._delegation_provider(_node, wait_timeout))
            _add_parent_loop(maf_kwargs, _node)
        message_node = top if top is not None else (_node or _message_node)
        if message_node is not None:
            from .delegation import MessagingProvider
            # Offers nothing until the session has a spawn policy (checked at each run).
            msg_kw = {} if wait_timeout is None else {"wait_timeout_s": wait_timeout}
            providers.append(MessagingProvider(self.delegation, message_node, **msg_kw))
        builtin = self.tools(*bundles)
        external: list = []
        for tool in tools or []:
            if looks_like_mcp(tool):
                external.extend(await self._attach_mcp(tool, agent=_mcp_agent))
            else:
                external.append(spill_tool(self, tool))
        external.extend(_prebuilt_tools or [])

        _apply_tool_loop_limits(client, self.config)
        if _node is not None:
            # The tools it may hand down, by name: the objects it was itself given.
            _node.tool_objects = {t.name: t for t in external}
        if top is not None:
            # The tools this agent may hand down, by name: the objects it was itself given.
            top.tool_objects = {t.name: t for t in external}
            if spawn is not None:
                from .delegation import Limits
                top.limits = Limits(timeout_s=self.config.subagent_task_timeout_s,
                                    max_round_trips=_round_trips(client),
                                    subtree_allowance=spawn.max_subagents,
                                    remaining_depth=spawn.max_depth)
        # Announce staged input trees on every run (a tree staged between turns included).
        maf_kwargs["context_providers"] = [*(maf_kwargs.get("context_providers") or []),
                                           _input_trees.context_provider(self), *providers]
        maf_kwargs.setdefault("max_context_window_tokens", self.config.max_context_window_tokens)
        maf_kwargs.setdefault("max_output_tokens", self.config.max_output_tokens)
        _default_compaction(maf_kwargs, self.config.keep_last_tool_call_groups)
        # The framework wires a tool-approval middleware by default that raises unless every
        # run() is given an AgentSession. We have no approval flow, and the returned agent is
        # public API, so keep it off: callers can run() it bare.
        maf_kwargs.setdefault("disable_tool_auto_approval", True)
        # The built-in OpenAI client stores conversations at OpenAI by default. This agent sends
        # fetched web content, documents and workspace data, so keep history local unless the
        # caller opts in with default_options={"store": True}. Merge, don't replace: the caller's
        # other chat options must survive. Connectors without the option are not sent it (some
        # reject unknown options), and {"store": None} drops it for any client: the framework
        # leaves None options out of the request.
        options = dict(maf_kwargs.get("default_options") or {})
        if "store" not in options and not _lacks_store_option(client):
            options["store"] = False
        maf_kwargs["default_options"] = options
        harness_instructions = self.sheath_instructions(*bundles)
        if _harness_extra:
            harness_instructions = f"{harness_instructions}\n\n{_harness_extra}"
        return create_harness_agent(
            client,
            name=name,
            harness_instructions=harness_instructions,
            agent_instructions=agent_instructions,
            tools=builtin + external,
            disable_todo=True,
            disable_mode=True,
            disable_file_memory=True,
            disable_web_search=True,
            **maf_kwargs,
        )

    def _set_spawn_policy(self, spawn: "SpawnPolicy",
                          client_factory: Callable[[str], Any] | None, *,
                          check_only: bool = False) -> None:
        """One policy and factory per session: its spawn budget is one per conversation.
        With ``check_only`` it raises on a conflict and sets nothing."""
        d = self.delegation
        # The policy by value (a frozen dataclass: an equal one is the same policy); the
        # factory by identity.
        if d.policy is not None and (d.policy != spawn
                                     or d.client_factory is not client_factory):
            raise ValueError("this session already has a different SpawnPolicy or "
                             "client_factory; a session takes one, shared by every agent "
                             "built on it")
        if not check_only:
            d.policy, d.client_factory = spawn, client_factory

    def _delegation_provider(self, top: Any, wait_timeout: Any) -> Any:
        from .delegation import DelegationProvider

        provider_kw = {} if wait_timeout is None else {"wait_timeout_s": wait_timeout}
        return DelegationProvider(self.delegation, top, **provider_kw)

    async def _build_subagents(self, client: Any, subagents: Mapping[str, "SubagentSpec"], *,
                               top: Any, bundles: tuple[str, ...],
                               client_factory: Callable[[str], Any] | None,
                               spawn: "SpawnPolicy | None" = None) -> None:
        """Validate the specs and build each sub-agent over this session as a node under
        ``top``. A spec with ``can_spawn`` gets ``spawn_subagent`` when there is a policy whose
        ``max_depth`` leaves a level beneath it."""
        from .api import expand_tools  # local: api imports session
        from .bundles import subagent_instructions
        from .delegation import AgentNode, Limits
        from .subagents import ConfinedSubagent, _ToolActivity, validate_subagents

        clients = validate_subagents(subagents, parent_bundles=bundles, parent_client=client,
                                     client_factory=client_factory, config=self.config)
        delegation = self.delegation
        taken = [n for n in subagents if delegation.node(n) is not None]
        if taken:
            raise ValueError(f"sub-agent names {taken} are already used in this session")
        depth_left = spawn.max_depth - 1 if spawn is not None else 0
        for sub_name, spec in subagents.items():
            sub_client = clients[sub_name]
            sub_bundles = spec.bundles or bundles
            can_spawn = bool(spec.can_spawn) and spawn is not None and depth_left >= 1
            node = AgentNode(name=sub_name, path=sub_name, depth=1, creator=top,
                             description=spec.description,
                             bundles=_bundles.selected_bundles(tuple(sub_bundles)),
                             can_spawn=can_spawn,
                             limits=Limits(timeout_s=self.config.subagent_task_timeout_s,
                                           max_round_trips=spec.max_round_trips,
                                           subtree_allowance=None,
                                           remaining_depth=depth_left if can_spawn else 0),
                             declared=True, client=sub_client, folder=f"agents/{sub_name}",
                             message_to=set(spec.message_to))
            agent = await self.create_agent(
                sub_client, agent_instructions=spec.instructions,
                tools=expand_tools(spec.tools), bundles=sub_bundles,
                name=sub_name, _mcp_agent=sub_name, _harness_extra=subagent_instructions(sub_name),
                _node=node if can_spawn else None, _message_node=node,
                middleware=[_ToolActivity()])
            # The spec's budget, on the sub-agent's own client only.
            _apply_tool_loop_limits(sub_client, dataclasses.replace(
                self.config, tool_loop_max_iterations=spec.max_round_trips,
                tool_loop_max_consecutive_errors=None))
            if can_spawn:
                # What it runs with: the ceiling for the sub-agents it creates.
                node.limits.max_round_trips = _round_trips(sub_client)
            with self.file_io():
                safe_path(self.root, node.folder).mkdir(parents=True, exist_ok=True)
            node.agent = ConfinedSubagent(agent, node=node, session=self)
            delegation.add_node(node)
        self.subagent_names = tuple(subagents)

    async def _attach_mcp(self, tool: Any, *, agent: str | None = None) -> list:
        """Connect an MCP server, attach the spill parser, capture its status, own its lifecycle.

        ``agent`` tags the server's status events with the sub-agent that owns it.
        """
        from .mcp_status import inject_progress_tokens, install_status_wrappers
        from .spill import make_spill_parser

        server = getattr(tool, "name", None) or repr(tool)
        _drop_mcp_host_payload(tool)
        # Install before connect so the wrappers reach the underlying MCP ClientSession.
        token_map = install_status_wrappers(self.status_bus, tool, server, agent=agent)
        try:
            await tool.connect()
        except Exception as e:  # noqa: BLE001 - add context naming the server, then re-raise
            raise RuntimeError(f"failed to connect MCP server {tool!r}: {e}") from e
        # Register before the parser loop so aclose() still closes this server if the loop raises.
        self._mcp_connected.append(tool)
        functions = list(tool.functions)
        # Wrap every function the model can reach, including ones it loads later with load_tool.
        every = _every_mcp_function(tool, functions)
        inject_progress_tokens(tool, server, token_map, every)   # after connect: tools are now loaded
        for ft in every:
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


# MAF keeps a raw copy of every MCP result in the result content's additional_properties, up
# to max_host_payload_size_bytes (1 MiB by default). The model never sees it, but compaction's
# token estimate counts it, so a result spilled to a short handle summary still weighed as
# much as the raw page. 1 byte is the smallest limit MAF accepts: every copy is then omitted.
_MCP_DEFAULT_HOST_PAYLOAD_BYTES = 1024 * 1024
_MCP_MIN_HOST_PAYLOAD_BYTES = 1


def _drop_mcp_host_payload(tool: Any) -> None:
    """Keep no raw MCP result copies, unless the caller chose a limit of their own."""
    if getattr(tool, "max_host_payload_size_bytes", None) != _MCP_DEFAULT_HOST_PAYLOAD_BYTES:
        return   # absent (not a MAF MCP tool), unlimited (None) or set by the caller
    tool.max_host_payload_size_bytes = _MCP_MIN_HOST_PAYLOAD_BYTES
    _quiet_mcp_omission_warnings()


class _MinLimitOmissionToDebug(logging.Filter):
    """Downgrade MAF's per-call 'Omitting MCP ...' warning when it reports sheath's own limit.

    At a 1-byte limit every result is omitted, so the warning would fire on every MCP call.
    A limit a caller chose still warns as usual.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if (record.levelno == logging.WARNING
                and str(record.msg).startswith("Omitting MCP")
                and record.args == (_MCP_MIN_HOST_PAYLOAD_BYTES,)):
            record.levelno, record.levelname = logging.DEBUG, "DEBUG"
            return logging.getLogger(record.name).isEnabledFor(logging.DEBUG)
        return True


def _quiet_mcp_omission_warnings() -> None:
    mcp_logger = logging.getLogger("agent_framework._mcp")
    if not any(isinstance(f, _MinLimitOmissionToDebug) for f in mcp_logger.filters):
        mcp_logger.addFilter(_MinLimitOmissionToDebug())


def _every_mcp_function(tool: Any, visible: list) -> list:
    """``visible`` plus, under MAF progressive disclosure, every allowed function loadable later.

    ``load_tool`` adds the very objects ``_filtered_functions()`` holds, so wrapping those at
    attach time covers them. Fails closed if that seam is gone: a deferred tool would otherwise
    bypass the spill parser silently.
    """
    if not getattr(tool, "use_progressive_disclosure", False):
        return visible
    filtered = getattr(tool, "_filtered_functions", None)
    if filtered is None:
        raise RuntimeError(
            f"MCP server {tool!r} uses progressive disclosure, but this agent-framework version has "
            "no _filtered_functions(); deferred tools could not be given the spill parser")
    seen = {id(ft) for ft in visible}
    return visible + [ft for ft in filtered() if id(ft) not in seen]


def _resolve_root(config: SheathConfig) -> tuple[Path, bool]:
    """Return ``(root, pre_existing)``, creating the root.

    A pinned ``root_dir`` is used verbatim. Otherwise a fresh directory is created under
    ``./.sheath/runs/`` with ``mkdtemp``, which is atomic: two threads or processes can
    never be handed the same root. That base is deliberately not ``./.sheath/sessions/``,
    where conversation roots are named by a caller-supplied id.
    """
    if config.root_dir is not None:
        root = Path(config.root_dir).resolve()
        pre_existing = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        return root, pre_existing
    base = Path.cwd() / ".sheath" / "runs"
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


def _make_tree_writable(top: Path, *, best_effort: bool = False) -> None:
    """Make every real directory under ``top`` listable and writable so rmtree can empty it.

    Walks top-down without following links (``os.walk`` default), chmods each directory
    before descending into it, and touches nothing that ``lstat`` doesn't show to be a real
    directory. Strict by default: the caller has already ``lstat``-ed ``top``, and any error
    propagates. With ``best_effort``, ``top`` is ``lstat``-checked here too, and an ``OSError``
    on any one entry is skipped (the rmtree that follows reports what is then left).
    """
    if best_effort:
        try:
            if not stat.S_ISDIR(os.lstat(top).st_mode):
                return
        except OSError:
            return
        try:
            os.chmod(top, 0o700)   # lstat just proved it is a real directory
        except OSError:
            pass
    else:
        os.chmod(top, 0o700)       # the caller's lstat proved it is a real directory
    for dirpath, dirnames, _ in os.walk(top):
        for d in dirnames:
            child = os.path.join(dirpath, d)
            try:
                if stat.S_ISDIR(os.lstat(child).st_mode):
                    os.chmod(child, 0o700)
            except OSError:
                if not best_effort:
                    raise


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


# Connector packages whose chat clients have no `store` option. The Anthropic connector forwards
# every non-None option to its SDK, which rejects `store`.
_NO_STORE_CONNECTORS = ("agent_framework_anthropic",)


def _lacks_store_option(client: Any) -> bool:
    """Whether ``client``'s class, or any base class, comes from a connector without `store`."""
    return any((cls.__module__ or "").split(".")[0] in _NO_STORE_CONNECTORS
               for cls in type(client).__mro__)


def _default_compaction(maf_kwargs: dict[str, Any], keep_last_tool_call_groups: int) -> None:
    """Fill in the compaction strategies the caller did not pass, in place.

    Mirrors the framework's own default (one context-window strategy shared by the per-call
    and post-turn phases) except that truncation never drops the first user message: in a
    one-shot task that message is the task, and a long run would go on without it.
    """
    if maf_kwargs.get("disable_compaction"):
        return
    if (maf_kwargs.get("before_compaction_strategy") is not None
            and maf_kwargs.get("after_compaction_strategy") is not None):
        return
    window, output = maf_kwargs.get("max_context_window_tokens"), maf_kwargs.get("max_output_tokens")
    if window is None or output is None:
        return  # no budget: the framework builds no token strategy either
    from agent_framework import ContextWindowCompactionStrategy

    strategy = ContextWindowCompactionStrategy(
        max_context_window_tokens=window, max_output_tokens=output,
        tokenizer=maf_kwargs.get("tokenizer"),
        keep_last_tool_call_groups=keep_last_tool_call_groups,
        preserve_first_user_group=True,
    )
    for phase in ("before_compaction_strategy", "after_compaction_strategy"):
        if maf_kwargs.get(phase) is None:
            maf_kwargs[phase] = strategy


def _add_parent_loop(maf_kwargs: dict[str, Any], node: Any) -> None:
    """Re-run a delegating agent while a task it started is still running, in place.

    Built here rather than through ``loop_should_continue``, which returns every pass joined
    together: ``return_final_only`` keeps only the final answer. A caller who passed
    ``loop_should_continue`` has taken over that decision, so nothing is added. The loop goes
    first in ``middleware``, ahead of the caller's own, preceded only by ``_DropLoopNudges``,
    which keeps the loop's nudges out of a streamed reply. ``node`` is the delegating agent's
    ``AgentNode``: the loop reads its tasks.
    """
    if "loop_should_continue" in maf_kwargs:
        return
    from agent_framework import AgentLoopMiddleware

    from .delegation import tasks_running, tasks_running_message
    from .subagents import _DropLoopNudges

    loop_kw: dict[str, Any] = {}
    max_iterations = maf_kwargs.pop("loop_max_iterations", None)
    if max_iterations is not None:
        loop_kw["max_iterations"] = max_iterations   # else MAF's default cap
    loop = AgentLoopMiddleware(tasks_running(node), next_message=tasks_running_message(node),
                               return_final_only=True, inject_progress=False, **loop_kw)
    # The nudge filter sits outside the loop, so it sees every pass's updates.
    maf_kwargs["middleware"] = [_DropLoopNudges(), loop, *(maf_kwargs.get("middleware") or [])]


# MAF's tool-loop round-trip cap when neither the host nor the client sets one.
_MAF_DEFAULT_ROUND_TRIPS = 40


def _round_trips(client: Any) -> int:
    """The round-trip budget ``client`` runs with (after ``_apply_tool_loop_limits``)."""
    settings = getattr(client, "function_invocation_configuration", None)
    value = settings.get("max_iterations") if isinstance(settings, dict) else None
    return value if isinstance(value, int) else _MAF_DEFAULT_ROUND_TRIPS


def _apply_tool_loop_limits(client: Any, config: SheathConfig) -> None:
    """Set the configured tool-loop limits on ``client``, where MAF reads them.

    MAF has no per-agent or per-request form of these limits, only the client's
    ``function_invocation_configuration``. A limit left at None is not touched, so a host's
    own setting on the client survives.
    """
    limits = {"max_iterations": ("tool_loop_max_iterations", 1),
              "max_consecutive_errors_per_request": ("tool_loop_max_consecutive_errors", 0)}
    chosen = {key: getattr(config, field) for key, (field, _) in limits.items()
              if getattr(config, field) is not None}
    for key, value in chosen.items():
        field, least = limits[key]
        if value < least:
            raise ValueError(f"SheathConfig.{field} must be at least {least}, got {value}")
    if not chosen:
        return
    settings = getattr(client, "function_invocation_configuration", None)
    if not isinstance(settings, dict):
        raise ValueError(f"{type(client).__name__} has no function_invocation_configuration, "
                         "so the SheathConfig tool-loop limits cannot be applied to it")
    settings.update(chosen)


def _build_sandbox(root: Path, store: HandleStore, sandbox_config: SandboxConfig,
                   limits: ControlPlaneLimits | None = None) -> SandboxExecutor:
    """Pick the sandbox backend from config (default: container). The one place the backend
    is decided; nothing else validates or selects it.

    Four guards, in order: (1) the isolation requirement -- ``require_isolation=True``
    refuses anything but the container tier, however it was selected (an explicit
    ``backend="local"`` or ``SHEATH_SANDBOX_BACKEND=local``); (2) the backend name, which
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
            f"SHEATH_SANDBOX_BACKEND selects the tier, not the container runtime -- "
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
                f'SheathConfig.sandbox.backend = "local" (or SHEATH_SANDBOX_BACKEND=local) '
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
