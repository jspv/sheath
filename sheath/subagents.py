"""Sub-agents: specs, their up-front validation, and the confining wrapper MAF's provider runs.

``Session.create_agent`` builds each sub-agent over its own ``Session`` and hands it to MAF's
``BackgroundAgentsProvider`` wrapped in ``ConfinedSubagent``, which bounds, tags and spills
every run.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_framework import AgentResponse, FunctionMiddleware, Message

from sheath.bundles import selected_bundles
from sheath.config import SheathConfig
from sheath.status import StatusEvent, bind_agent, report_progress

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_NAME_PATTERN = "^[A-Za-z][A-Za-z0-9_-]{0,63}$"


@dataclass(frozen=True, kw_only=True)
class SubagentSpec:
    description: str                         # what it is for; shown to the parent model
    instructions: str | None = None          # its domain prompt (agent_instructions)
    bundles: tuple[str, ...] | None = None   # None = the parent's selection
    tools: tuple | list = ()                 # callables, ToolFactory, MCP tools
    model: str | None = None                 # model for a client sheath builds; None = config.model
    client: Any | None = None                # a ready client of its own, instead of model
    max_round_trips: int | None = None       # per-task tool-loop budget


def validate_subagents(
    specs: Mapping[str, SubagentSpec],
    *,
    parent_bundles: tuple[str, ...],
    parent_client: Any,
    client_factory: Callable[[str], Any] | None,
    config: SheathConfig,
) -> dict[str, Any]:
    """Check every spec, then resolve each one's own client: ``{name: client}`` in order.

    All errors are raised before ``client_factory`` is called for any spec.
    """
    if config.subagent_max_concurrent < 1:
        raise ValueError(
            f"subagent_max_concurrent must be >= 1, got {config.subagent_max_concurrent}"
        )
    if config.subagent_task_timeout_s is not None and config.subagent_task_timeout_s <= 0:
        raise ValueError(
            f"subagent_task_timeout_s must be > 0 or None, got {config.subagent_task_timeout_s}"
        )
    parent_allowed = set(selected_bundles(parent_bundles))
    seen_names: dict[str, str] = {}
    seen_clients: list[Any] = []
    for name, spec in specs.items():
        if not isinstance(spec, SubagentSpec):
            raise TypeError(
                f"subagents[{name!r}] must be a SubagentSpec, got {type(spec).__name__}"
            )
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(f"sub-agent name {name!r} must match {_NAME_PATTERN}")
        folded = name.lower()
        if folded in seen_names:
            raise ValueError(
                f"sub-agent names {seen_names[folded]!r} and {name!r} are equal ignoring case"
            )
        seen_names[folded] = name
        if spec.bundles is not None:
            outside = [b for b in selected_bundles(spec.bundles) if b not in parent_allowed]
            if outside:
                raise ValueError(
                    f"sub-agent {name!r} asks for bundles {outside} outside the parent's "
                    f"selection {sorted(parent_allowed)}"
                )
        if spec.max_round_trips is not None and spec.max_round_trips < 1:
            raise ValueError(
                f"sub-agent {name!r}: max_round_trips must be >= 1, got {spec.max_round_trips}"
            )
        if spec.model is not None and spec.client is not None:
            raise ValueError(f"sub-agent {name!r}: set model or client, not both")
        if spec.client is not None:
            if spec.client is parent_client:
                raise ValueError(
                    f"sub-agent {name!r}: client is the parent's client; give it one of its own"
                )
            if any(spec.client is c for c in seen_clients):
                raise ValueError(
                    f"sub-agent {name!r}: client is shared with another sub-agent; "
                    "each needs its own"
                )
            seen_clients.append(spec.client)
        elif client_factory is None:
            raise ValueError(
                f"sub-agent {name!r} has no client: pass client= on the SubagentSpec "
                "or a client_factory"
            )
    resolved: dict[str, Any] = {}
    for name, spec in specs.items():
        resolved[name] = (
            spec.client
            if spec.client is not None
            else client_factory(spec.model or config.model)  # type: ignore[misc]
        )
    return resolved


_PROVIDER_NOTE = (
    "Sub-agents work in this workspace under the same limits as you. Delegate a self-contained "
    "side task with background_agents_start_task, wait for every task you start before you "
    "finish, and open a long result with inspect_handle."
)


def provider_instructions() -> str:
    """sheath's note on delegation, followed by MAF's listing of the sub-agents."""
    return _PROVIDER_NOTE + "\n\n{background_agents}"


class _ToolActivity(FunctionMiddleware):
    """Emit ``running`` before each sub-agent tool call and ``done`` or ``failed: <type>`` after.

    The agent tag bound by ``ConfinedSubagent.run`` fills in ``agent``. The exception message is
    left out on purpose: it can echo untrusted content, and the delegating agent already gets
    the full error through the tool result.
    """

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        name = context.function.name
        report_progress("running", tool=name)
        try:
            await call_next()
        except BaseException as e:
            report_progress(f"failed: {type(e).__name__}", tool=name)
            raise
        report_progress("done", tool=name)


class ConfinedSubagent:
    """What MAF's ``BackgroundAgentsProvider`` runs: a sheath-built agent, bounded per run.

    Each run is registered with the session (so it can be cancelled, queued or running), waits
    for one of the session's ``subagent_max_concurrent`` slots, is tagged with the sub-agent's
    name for status, is bounded by ``subagent_task_timeout_s``, and has its report spilled to a
    handle when it is large.
    """

    def __init__(self, inner: Any, *, name: str, description: str, session: Any) -> None:
        self._inner = inner
        self.name = name
        self.description = description
        self._session = session

    @property
    def id(self) -> str:
        return self._inner.id

    def create_session(self, **kwargs: Any) -> Any:
        return self._inner.create_session(**kwargs)

    def get_session(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.get_session(*args, **kwargs)

    async def run(self, messages: Any = None, *, stream: bool = False, session: Any = None,
                  **kwargs: Any) -> AgentResponse:
        from sheath.spill import _maybe_spill  # local: spill imports session -> circular

        if stream:
            raise ValueError(f"sub-agent {self.name!r} does not support stream=True")
        sess = self._session
        task = asyncio.current_task()
        if task is not None:
            sess._subagent_tasks.add(task)
        try:
            if sess._subagent_slots is None:
                # Created here, not with the session: it must belong to the running loop.
                sess._subagent_slots = asyncio.Semaphore(sess.config.subagent_max_concurrent)
            async with sess._subagent_slots:
                with bind_agent(self.name):
                    return await self._run_tagged(messages, session, kwargs, _maybe_spill)
        finally:
            if task is not None:
                sess._subagent_tasks.discard(task)

    async def _run_tagged(self, messages: Any, session: Any, kwargs: dict[str, Any],
                          maybe_spill: Callable[..., Any]) -> AgentResponse:
        sess = self._session
        sess.status_bus.emit(StatusEvent(tool="sheath", message="started", agent=self.name))
        try:
            timeout = sess.config.subagent_task_timeout_s
            bound = asyncio.timeout(timeout)   # None: no bound
            try:
                async with bound:
                    response = await self._inner.run(messages, session=session, **kwargs)
            except TimeoutError:
                if not bound.expired():
                    raise   # the agent's own TimeoutError, not our bound
                raise TimeoutError(
                    f"sub-agent {self.name!r} ran past subagent_task_timeout_s ({timeout:g}s)"
                ) from None
            report = maybe_spill(sess, self.name, response.text,
                                 source=f"subagent:{self.name}")
            if isinstance(report, dict):
                report = json.dumps(report)
        except BaseException as e:
            sess.status_bus.emit(StatusEvent(tool="sheath", message=f"failed: {type(e).__name__}",
                                             agent=self.name))
            raise
        sess.status_bus.emit(StatusEvent(tool="sheath", message="finished", agent=self.name))
        return AgentResponse(messages=[Message(role="assistant", contents=[report])])
