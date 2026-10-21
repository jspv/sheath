"""SessionManager: an in-process registry of continuous Conversations, keyed by id.

Open-or-create by id (e.g. an AG-UI threadId), get, close, lazy TTL expiry, and a sweep the host
can call on its own cadence. Each conversation gets an isolated per-id root so concurrent
conversations never collide. The ConversationStore seam lets a disk-backed impl drop in later.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from .conversation import Conversation
from .paths import safe_path, validate_segment


class ConversationStore(Protocol):
    def get(self, conv_id: str) -> Conversation | None: ...
    def put(self, conv_id: str, conv: Conversation) -> None: ...
    def pop(self, conv_id: str) -> Conversation | None: ...
    def items(self) -> list[tuple[str, Conversation]]: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._d: dict[str, Conversation] = {}

    def get(self, conv_id: str) -> Conversation | None:
        return self._d.get(conv_id)

    def put(self, conv_id: str, conv: Conversation) -> None:
        self._d[conv_id] = conv

    def pop(self, conv_id: str) -> Conversation | None:
        return self._d.pop(conv_id, None)

    def items(self) -> list[tuple[str, Conversation]]:
        return list(self._d.items())


def _conv_root(config: Any, conv_id: str) -> Path:
    """Isolated per-conversation root. Uses config.root_dir as a base, else ./.sheath/sessions.

    ``conv_id`` may be untrusted (e.g. an AG-UI ``threadId``). It must name a *single* child of the
    base — a separator or ``..`` would let it nest under another conversation or escape the base
    entirely, and on close it would be ``rmtree``'d (see ``Conversation.aclose``). We reject
    separators up front and run the result through ``safe_path`` as a containment backstop.
    """
    base = Path(config.root_dir) if config.root_dir else (Path.cwd() / ".sheath" / "sessions")
    validate_segment(conv_id, what="conversation id")
    return safe_path(base, conv_id)


class SessionManager:
    def __init__(self, sheath: Any, *, idle_ttl_s: float | None = None,
                 store: ConversationStore | None = None) -> None:
        self._sheath = sheath
        self._ttl = idle_ttl_s
        self._store: ConversationStore = store or InMemoryConversationStore()
        self._lock = asyncio.Lock()
        # Ids whose popped conversation is still tearing down; aopen of such an id waits.
        self._reaping: dict[str, asyncio.Event] = {}

    def _expired(self, conv: Conversation) -> bool:
        # A conversation with a turn in flight is by definition not idle.
        if self._ttl is None or conv.busy:
            return False
        return (time.monotonic() - conv.last_activity) > self._ttl

    async def aopen(self, session_id: str | None = None, *, tools: list | None = None,
                    bundles: tuple[str, ...] | None = None,
                    agent_instructions: str | None = None,
                    on_publish: Any | None = None,
                    agent_kwargs: dict[str, Any] | None = None) -> Conversation:
        """Reuse the live conversation for ``session_id`` (if any, not expired), else create one.

        Open-or-create is serialized behind a single manager lock so a concurrent ``aopen`` of the
        same id returns the same Conversation rather than racing two builds. The lock is held across
        ``Conversation.acreate`` (which connects MCP), so a slow connect for one id delays other
        opens — acceptable at v1's scale; a per-id lock is the fast-follow if it bites. Reaping
        an expired conversation (whose teardown may wait for a running turn) happens outside
        the manager lock; only opens of that same id wait for it.

        ``agent_instructions``, ``on_publish`` and ``agent_kwargs`` override the Sheath defaults for
        a conversation being *created*; re-opening a live id returns the existing conversation unchanged.
        """
        while True:
            async with self._lock:
                reaping = self._reaping.get(session_id) if session_id is not None else None
                if reaping is None:
                    existing = (self._store.get(session_id) if session_id is not None
                                else None)
                    if existing is None:
                        return await self._create(session_id, tools=tools, bundles=bundles,
                                                  agent_instructions=agent_instructions,
                                                  on_publish=on_publish,
                                                  agent_kwargs=agent_kwargs)
                    if not self._expired(existing):
                        # Opening is activity: without this, a caller could be handed a
                        # conversation that a sweep reaps before its turn takes the lock.
                        existing.last_activity = time.monotonic()
                        return existing
                    # Expired: reap it, but outside the manager lock (its teardown may wait
                    # up to 30 s for a turn). The reservation makes opens of this id wait.
                    self._store.pop(session_id)
                    done = self._reserve(session_id)
            if reaping is not None:
                await reaping.wait()          # another task is tearing this id down
            else:
                await self._reap(session_id, existing, done)
            # Retry: the id is now free (or someone else recreated it first).

    async def _create(self, session_id: str | None, *, tools, bundles, agent_instructions,
                      on_publish, agent_kwargs) -> Conversation:
        conv_id = session_id if session_id is not None else uuid.uuid4().hex
        config = dataclasses.replace(
            self._sheath.config, root_dir=str(_conv_root(self._sheath.config, conv_id)))
        from .api import expand_tools

        conv = await Conversation.acreate(
            id=conv_id, config=config, client=self._sheath._make_client(),
            tools=expand_tools(self._sheath._tools) + expand_tools(tools),
            bundles=bundles if bundles is not None else self._sheath._bundles,
            reap_on_close=True,
            agent_instructions=(agent_instructions if agent_instructions is not None
                                else self._sheath._agent_instructions),
            on_publish=(on_publish if on_publish is not None
                        else self._sheath._on_publish),
            agent_kwargs=self._sheath._merged_agent_kwargs(agent_kwargs),
        )
        self._store.put(conv_id, conv)
        return conv

    def _reserve(self, conv_id: str) -> asyncio.Event:
        """Mark ``conv_id`` as being torn down. Call synchronously right after popping it
        from the store, so no ``aopen`` can create a replacement whose root (the same
        per-id directory) the old conversation's teardown would then delete."""
        done = asyncio.Event()
        self._reaping[conv_id] = done
        return done

    async def _reap(self, conv_id: str, conv: Conversation, done: asyncio.Event) -> None:
        try:
            await conv.aclose()
        finally:
            if self._reaping.get(conv_id) is done:
                del self._reaping[conv_id]
            done.set()

    def get(self, session_id: str) -> Conversation | None:
        """Live conversation for ``session_id``, or None if absent/expired.

        Lazily *hides* an expired conversation but does not reap it — reaping happens on the next
        ``aopen`` of that id or in ``sweep``. A host that never re-opens expired ids must call
        ``sweep`` on a cadence to bound disk/MCP-connection use.
        """
        conv = self._store.get(session_id)
        if conv is None or self._expired(conv):
            return None
        return conv

    async def close(self, session_id: str) -> None:
        conv = self._store.pop(session_id)
        if conv is not None:
            await self._reap(session_id, conv, self._reserve(session_id))

    async def sweep(self) -> None:
        for conv_id, conv in self._store.items():
            # Identity check: an earlier reap's await may have let this id be recreated.
            if self._expired(conv) and self._store.get(conv_id) is conv:
                self._store.pop(conv_id)
                await self._reap(conv_id, conv, self._reserve(conv_id))

    async def aclose(self) -> None:
        for conv_id, conv in self._store.items():
            if self._store.get(conv_id) is not conv:
                continue
            self._store.pop(conv_id)
            await self._reap(conv_id, conv, self._reserve(conv_id))
