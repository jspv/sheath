"""Conversation: a stateful, multi-turn handle over a persistent workspace.

It keeps one workspace ``Session`` open across turns (so handles persist) and reuses one MAF
``AgentSession`` (so conversation history threads, MAF-managed). ``aask`` runs one turn under a
single-flight lock; ``aclose`` tears down MCP/bus and reaps the workspace unless ``reap_on_close``
is False. Async-first: a persistent agent + MCP connections require one stable event loop.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from .config import SheathConfig
from .publish import OnPublish
from .session import Session, _remove_root_locked

if TYPE_CHECKING:
    from pathlib import Path

    from .api import Result
    from collections.abc import Callable

    from .handles import Handle
    from .input_trees import InputTree


_log = logging.getLogger(__name__)

# How long ``aclose`` waits for a running turn before tearing down anyway.
_CLOSE_WAIT_S = 30.0
_CLOSED = "conversation is closed"


class Conversation:
    def __init__(self, conv_id: str, session: Session, agent: Any, agent_session: Any,
                 *, reap_on_close: bool = True) -> None:
        self.id = conv_id
        self.session = session
        self.agent = agent
        self.agent_session = agent_session
        self.reap_on_close = reap_on_close
        self.last_activity = time.monotonic()
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def acreate(cls, *, id: str, config: SheathConfig, client: Any,
                      tools: list | None = None, bundles: tuple[str, ...] = ("code", "files", "web"),
                      reap_on_close: bool = True,
                      agent_instructions: str | None = None,
                      on_publish: OnPublish | None = None,
                      agent_kwargs: dict[str, Any] | None = None,
                      subagents: Any | None = None,
                      spawn: Any | None = None,
                      client_factory: Any | None = None,
                      subagent_middleware: Any | None = None) -> "Conversation":
        """Open the workspace, build the agent once, start a MAF conversation thread.

        ``agent_instructions`` (the host's domain system prompt) is appended after sheath's
        operating manual by MAF: ``<core + bundle instructions>\\n\\n<agent_instructions>``.
        ``agent_kwargs`` are forwarded to ``Session.create_agent``; ``subagents``, ``spawn``,
        ``client_factory`` and ``subagent_middleware`` likewise, and win over the same keys in
        ``agent_kwargs`` when given.
        """
        extra = dict(agent_kwargs or {})
        if subagents is not None:
            extra["subagents"] = subagents
        if spawn is not None:
            extra["spawn"] = spawn
        if client_factory is not None:
            extra["client_factory"] = client_factory
        if subagent_middleware is not None:
            extra["subagent_middleware"] = subagent_middleware
        session = Session.create(config, on_publish=on_publish, bundles=bundles)
        await session.__aenter__()
        try:
            agent = await session.create_agent(client, agent_instructions=agent_instructions,
                                                tools=tools or [], bundles=bundles,
                                                **extra)
            agent_session = agent.create_session(session_id=id)
        except BaseException:
            await session.__aexit__(None, None, None)
            raise
        return cls(id, session, agent, agent_session, reap_on_close=reap_on_close)

    async def aask(self, question: str) -> "Result":
        """Run one turn (serialized per conversation). Returns a Result; errors are captured."""
        from .api import Result

        async with self._lock:
            if self._closed:   # e.g. queued behind aclose(): the workspace is gone
                return Result(final_text="", handles={}, files=[],
                              session_dir=self.session.root, error=f"RuntimeError: {_CLOSED}")
            final_text, error = "", None
            try:
                response = await self.agent.run(question, session=self.agent_session)
                final_text = response.text
            except Exception as e:  # noqa: BLE001 - surface, don't crash the conversation
                error = f"{type(e).__name__}: {e}"
            finally:
                # Tasks belong to the turn: none keeps running once it ends.
                await self.session.cancel_subagents()
            self.last_activity = time.monotonic()
            return Result(final_text=final_text, handles=dict(self.session.handles),
                          files=self.session.artifacts, session_dir=self.session.root, error=error)

    @property
    def busy(self) -> bool:
        """True while a turn (``aask`` or an AG-UI stream) holds this conversation."""
        return self._lock.locked()

    async def astream_agui(self, input_data: dict, **agui_kwargs: Any) -> AsyncIterator[Any]:
        """Run one AG-UI turn: serialized with every other turn on this conversation, and
        counted as activity at its start and its end so an idle TTL measures real idleness.

        The lock is released when the stream is exhausted **or closed**: a server that
        closes this generator on client disconnect frees the conversation for the next turn.
        """
        from . import agui

        async with self._lock:
            if self._closed:
                raise RuntimeError(_CLOSED)
            self.last_activity = time.monotonic()
            try:
                async with aclosing(agui.agui_event_stream(
                        self.agent, self.session.status_bus, input_data,
                        **agui_kwargs)) as events:
                    async for event in events:
                        yield event
            finally:
                try:
                    # Tasks belong to the turn, also when the client disconnects mid-stream.
                    await self.session.cancel_subagents()
                finally:
                    self.last_activity = time.monotonic()

    @property
    def inputs(self) -> dict[str, "Handle"]:
        """User-provided inputs in this conversation's workspace, by ``input_id``."""
        return self.session.inputs

    def add_input(self, source: "Path | bytes", *, input_id: str, name: str,
                  content_type: str | None = None,
                  description: str | None = None) -> "Handle":
        """Ingest a user upload between turns (see ``Session.add_input``).

        Raises ``RuntimeError`` while a turn is running; use ``aadd_input`` to wait instead.
        """
        if self._closed:
            raise RuntimeError(_CLOSED)
        if self._lock.locked():
            raise RuntimeError("cannot add an input while a turn is running; use aadd_input()")
        return self.session.add_input(source, input_id=input_id, name=name,
                                      content_type=content_type, description=description)

    async def aadd_input(self, source: "Path | bytes", *, input_id: str, name: str,
                         content_type: str | None = None,
                         description: str | None = None) -> "Handle":
        """Ingest a user upload, waiting for any running turn to finish first. The copy runs
        in a worker thread so a large upload does not stall the event loop."""
        async with self._lock:
            if self._closed:
                raise RuntimeError(_CLOSED)
            return await asyncio.to_thread(
                self.session.add_input, source, input_id=input_id, name=name,
                content_type=content_type, description=description)

    @property
    def input_trees(self) -> dict[str, "InputTree"]:
        """Directory trees staged in this conversation's workspace, by ``input_id``."""
        return self.session.input_trees

    def add_input_tree(self, source: "Path | str", *, input_id: str,
                       description: str | None = None,
                       ignore: "Callable[[str], bool] | None" = None,
                       replace: bool = False) -> "InputTree":
        """Stage a directory read-only between turns (see ``Session.add_input_tree``).

        Raises ``RuntimeError`` while a turn is running; use ``aadd_input_tree`` to wait.
        """
        if self._closed:
            raise RuntimeError(_CLOSED)
        if self._lock.locked():
            raise RuntimeError("cannot add an input tree while a turn is running; "
                               "use aadd_input_tree()")
        return self.session.add_input_tree(source, input_id=input_id, description=description,
                                           ignore=ignore, replace=replace)

    async def aadd_input_tree(self, source: "Path | str", *, input_id: str,
                              description: str | None = None,
                              ignore: "Callable[[str], bool] | None" = None,
                              replace: bool = False) -> "InputTree":
        """Stage a directory read-only, waiting for any running turn to finish first. The copy
        runs in a worker thread so a large tree does not stall the event loop."""
        async with self._lock:
            if self._closed:
                raise RuntimeError(_CLOSED)
            return await asyncio.to_thread(
                self.session.add_input_tree, source, input_id=input_id,
                description=description, ignore=ignore, replace=replace)

    async def aclose(self) -> None:
        """Tear down the workspace (MCP + bus), then reap the root unless retained. Idempotent.

        Waits (up to ``_CLOSE_WAIT_S``) for a running turn to finish first, so the root is
        never removed under a turn that is still using it; turns queued behind the close see
        the conversation closed and do not run. If the wait times out, teardown proceeds
        anyway (logged), since a host closing a conversation must be able to reclaim it.
        """
        if self._closed:
            return
        self._closed = True   # mark closed up front: a teardown failure isn't retried
        acquired = False
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=_CLOSE_WAIT_S)
            acquired = True
        except TimeoutError:
            _log.warning("conversation %s: a turn was still running after %.0fs; closing "
                         "anyway", self.id, _CLOSE_WAIT_S)
        try:
            try:
                await self.session.release_subagents(self.agent_session)
            finally:
                await self.session.__aexit__(None, None, None)
        finally:
            try:
                if self.reap_on_close:
                    # In a worker thread, under io_lock: a sync file tool still in flight
                    # finishes before the directory goes.
                    await asyncio.to_thread(
                        _remove_root_locked, self.session.root, self.session.io_lock,
                        getattr(self.session.sandbox, "check_quarantine", None))
            except OSError:
                _log.warning("conversation %s: could not remove %s", self.id,
                             self.session.root, exc_info=True)
            finally:
                if acquired:
                    self._lock.release()
