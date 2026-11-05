"""Sub-agents: specs, their up-front validation, and the confining wrapper each task runs in.

``Session.create_agent`` builds each sub-agent over the same ``Session``, wraps it in
``ConfinedSubagent`` (which bounds, tags and spills every run) and registers it as an
``AgentNode`` with the session's ``Delegation`` (``sheath.delegation``).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_framework import (AgentMiddleware, AgentResponse, FunctionMiddleware, Message,
                             ResponseStream)

from sheath.bundles import selected_bundles
from sheath.config import SheathConfig
from sheath.delegation import CURRENT_RUN, RunState, attach, detach
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
    can_spawn: bool = False                  # give it spawn_subagent (needs a SpawnPolicy)
    message_to: tuple[str, ...] = ()         # agents it may message besides its creator


@dataclass(frozen=True, kw_only=True)
class SpawnPolicy:
    """The host's bounds on sub-agents the model creates at runtime with ``spawn_subagent``."""

    models: tuple[str, ...]          # allowlist; required, non-empty
    default_model: str | None = None # used when the agent names none; None = models[0]
    max_subagents: int = 8           # spawned per conversation, at every depth, ever
    max_depth: int = 3               # backstop on nesting: depth of the deepest sub-agent
    max_messages: int = 200          # send_message calls per conversation
    max_message_chars: int = 4000    # per message

    @property
    def effective_default(self) -> str:
        return self.default_model if self.default_model is not None else self.models[0]


def validate_spawn_policy(policy: SpawnPolicy, client_factory: Callable[[str], Any] | None
                          ) -> None:
    """Raise ``ValueError`` for a policy sheath cannot honour, or one without a factory."""
    if not isinstance(policy, SpawnPolicy):
        raise TypeError(f"spawn must be a SpawnPolicy, got {type(policy).__name__}")
    models = policy.models
    if not isinstance(models, tuple) or not models:
        raise ValueError("SpawnPolicy.models must be a non-empty tuple of model names")
    for m in models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"SpawnPolicy.models: every model must be a non-empty string, "
                             f"got {m!r}")
    if policy.default_model is not None and policy.default_model not in models:
        raise ValueError(f"SpawnPolicy.default_model {policy.default_model!r} is not in "
                         f"models {list(models)}")
    for field_name in ("max_subagents", "max_depth", "max_messages", "max_message_chars"):
        value = getattr(policy, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SpawnPolicy.{field_name} must be an integer >= 1, got {value!r}")
    if client_factory is None:
        raise ValueError("a SpawnPolicy needs a client_factory: each spawned sub-agent gets a "
                         "client of its own from client_factory(model)")


def client_reuse(client: Any, *, parent_client: Any, others: list[Any]) -> str | None:
    """Why ``client`` cannot be a new sub-agent's own client, or None when it can.

    A client shared with another agent shares its tool-loop budget: the last budget applied to
    it would govern them all, the parent's included."""
    if client is parent_client:
        return "the parent's client"
    if any(client is c for c in others):
        return "a client already given to another sub-agent"
    return None


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
        if (not isinstance(spec.message_to, (tuple, list))
                or not all(isinstance(m, str) and _NAME_RE.fullmatch(m)
                           for m in spec.message_to)):
            raise ValueError(
                f"sub-agent {name!r}: message_to must be a tuple of agent names matching "
                f"{_NAME_PATTERN}, got {spec.message_to!r}"
            )
        if spec.max_round_trips is not None and spec.max_round_trips < 1:
            raise ValueError(
                f"sub-agent {name!r}: max_round_trips must be >= 1, got {spec.max_round_trips}"
            )
        if spec.model is not None and spec.client is not None:
            raise ValueError(f"sub-agent {name!r}: set model or client, not both")
        if spec.client is not None:
            reused = client_reuse(spec.client, parent_client=parent_client, others=seen_clients)
            if reused == "the parent's client":
                raise ValueError(
                    f"sub-agent {name!r}: client is the parent's client; give it one of its own"
                )
            if reused is not None:
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
        if spec.client is not None:
            resolved[name] = spec.client
            continue
        made = client_factory(spec.model or config.model)  # type: ignore[misc]
        # A caching factory would hand one client to several agents.
        reused = client_reuse(made, parent_client=parent_client, others=seen_clients)
        if reused is not None:
            raise ValueError(
                f"sub-agent {name!r}: client_factory returned {reused}; "
                "it must return a new client for each sub-agent"
            )
        seen_clients.append(made)
        resolved[name] = made
    return resolved


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


class _DropLoopNudges(AgentMiddleware):
    """Keep the parent loop's nudges out of a streamed reply.

    Placed ahead of the loop middleware. On a streaming run the loop yields each nudge it sends
    the model as a ``role="user"`` update; a frontend would fold it into the assistant's text.
    This skips those updates and returns the inner stream's own final response. The model's
    text, including a premature "done" before it waits, is real output and stays. Non-streaming
    runs pass through.
    """

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        await call_next()
        if not context.stream:
            return
        inner = context.result

        async def updates() -> Any:
            try:
                async for update in inner:
                    if str(getattr(update, "role", "")) == "user":
                        continue
                    yield update
            except BaseException:
                await inner.close()        # abandoned or failed: release the inner run
                raise

        async def final(_updates: Any) -> Any:
            return await inner.get_final_response()

        context.result = ResponseStream(updates(), finalizer=final)


class ConfinedSubagent:
    """What a delegated task runs: a sheath-built agent, bounded per run.

    Each run is registered with the session (so it can be cancelled, queued or running), waits
    for one of the session's ``subagent_max_concurrent`` slots, is tagged with the sub-agent's
    path for status, is bounded by its node's effective ``timeout_s`` (the host's
    ``subagent_task_timeout_s`` unless its creator set a lower one), and has its report spilled
    to a handle when it is large. While it waits on its own sub-agents (``AgentNode.waiting``)
    it gives its slot back and its clock stops. When it ends, for any reason, every task it
    started that is still running is cancelled and awaited (the cascade).
    """

    def __init__(self, inner: Any, *, node: Any, session: Any) -> None:
        self._inner = inner
        self._node = node                  # its sheath.delegation.AgentNode
        self.name = node.path
        self.description = node.description
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
        # Messages for this sub-agent go into this run's model calls; its inbox first.
        attach(self._node, session)
        # This run, for AgentNode.waiting in its tool calls: the slot it holds, its clock.
        run = RunState(node=self._node)
        token = CURRENT_RUN.set(run)
        try:
            # Yield once right after registering, so Session.cancel_subagents (which yields
            # before it looks again) sees a run started outside the delegation's own records
            # that had not yet run, and cancels it before it can do any work.
            await asyncio.sleep(0)
            if sess._subagent_slots is None:
                # Created here, not with the session: it must belong to the running loop.
                sess._subagent_slots = asyncio.Semaphore(sess.config.subagent_max_concurrent)
            # Held by the run, not an ``async with``: waiting() hands it back and takes it again.
            run.slots = sess._subagent_slots
            await run.acquire()
            with bind_agent(self.name):
                return await self._run_tagged(messages, session, kwargs, _maybe_spill, run)
        finally:
            run.close()            # releases only a slot it holds; drops a pending re-acquire
            CURRENT_RUN.reset(token)
            try:
                # Nothing runs under a creator whose task is gone: cancel what this run started.
                delegation = self._node._delegation
                if delegation is not None:
                    await delegation.cascade(self._node, session)
            finally:
                detach(self._node, session, force=True)
                if task is not None:
                    sess._subagent_tasks.discard(task)

    async def _run_tagged(self, messages: Any, session: Any, kwargs: dict[str, Any],
                          maybe_spill: Callable[..., Any], run: RunState) -> AgentResponse:
        sess = self._session
        sess.status_bus.emit(StatusEvent(tool="sheath", message="started", agent=self.name))
        try:
            timeout = self._node.limits.timeout_s   # fixed when the node was created
            limit_name = ("subagent_task_timeout_s"
                          if timeout == sess.config.subagent_task_timeout_s else "timeout_s")
            bound = asyncio.timeout(timeout)   # None: no bound
            run.bound = bound                  # waiting() pauses it
            try:
                try:
                    async with bound:
                        response = await self._inner.run(messages, session=session, **kwargs)
                finally:
                    run.bound = None           # exited: no longer reschedulable
            except TimeoutError:
                if not bound.expired():
                    raise   # the agent's own TimeoutError, not our bound
                raise TimeoutError(
                    f"sub-agent {self.name!r} ran past {limit_name} ({timeout:g}s)"
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
