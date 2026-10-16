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
import shutil
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from .config import TetherConfig
from .publish import OnPublish
from .session import Session

if TYPE_CHECKING:
    from pathlib import Path

    from .api import Result
    from .handles import Handle


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
    async def acreate(cls, *, id: str, config: TetherConfig, client: Any,
                      tools: list | None = None, bundles: tuple[str, ...] = ("code", "files", "web"),
                      reap_on_close: bool = True,
                      agent_instructions: str | None = None,
                      on_publish: OnPublish | None = None) -> "Conversation":
        """Open the workspace, build the agent once, start a MAF conversation thread.

        ``agent_instructions`` (the host's domain system prompt) is appended after tether's
        operating manual by MAF: ``<core + bundle instructions>\\n\\n<agent_instructions>``.
        """
        session = Session.create(config, on_publish=on_publish, bundles=bundles)
        await session.__aenter__()
        try:
            agent = await session.create_agent(client, agent_instructions=agent_instructions,
                                                tools=tools or [], bundles=bundles)
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
            await self.session.__aexit__(None, None, None)
        finally:
            try:
                if self.reap_on_close and self.session.root.exists():
                    shutil.rmtree(self.session.root, ignore_errors=True)
            finally:
                if acquired:
                    self._lock.release()
