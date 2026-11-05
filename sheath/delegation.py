"""sheath's delegation provider: the agent tree, its tasks, and the ``background_agents_*`` tools.

One ``Delegation`` per ``Session`` (one per conversation) holds every agent node and every task,
so task state outlives any single MAF ``AgentSession``: a fresh ``AgentSession`` on the same
conversation (as AG-UI builds per request) still sees the tasks started earlier.

``DelegationProvider`` gives one agent node the six tools under MAF's names, with MAF's return
strings, so prompts, tests and host UIs written against MAF's provider keep working.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_framework import ContextProvider, Message, tool
from agent_framework._sessions import (
    _MESSAGE_INJECTION_LOCK,
    MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY,
    enqueue_messages,
)

_log = logging.getLogger(__name__)

DEFAULT_WAIT_TIMEOUT_S = 300
# How long cancelling tasks waits for them to finish before giving up (logged).
CANCEL_WAIT_S = 30.0
# How many times cancelling yields and looks again for newly registered runs.
CANCEL_ROUNDS = 3

_SOURCE_ID = "sheath_delegation"
# Marks a message ``deliver`` queued on an AgentSession, so an undrained one can be taken back.
_DELIVERY_KEY = "sheath_delivery"

_PROVIDER_NOTE = (
    "Sub-agents work in this workspace under the same limits as you. Delegate a self-contained "
    "side task with background_agents_start_task, wait for every task you start before you "
    "finish, and open a long result with inspect_handle."
)


@dataclass
class Limits:
    timeout_s: float | None
    max_round_trips: int | None
    subtree_allowance: int | None
    remaining_depth: int


@dataclass
class RunState:
    """One ``ConfinedSubagent`` run of ``node``: the slot it holds and the clock bounding it.

    Set as the current run (``CURRENT_RUN``) for the run's whole context, so its tool calls see
    it; ``AgentNode.waiting`` hands the slot back and pauses the clock while the run waits."""

    node: AgentNode
    slots: asyncio.Semaphore | None = None
    held: bool = False                       # True while this run holds one of ``slots``
    bound: asyncio.Timeout | None = None     # the run's subagent_task_timeout_s bound
    _waiters: int = field(default=0, repr=False)          # waits in progress in this run
    _wait_began: float = field(default=0.0, repr=False)    # loop time the first one began
    _wait_deadline: float | None = field(default=None, repr=False)  # the paused deadline

    async def acquire(self) -> None:
        assert self.slots is not None and not self.held
        await self.slots.acquire()
        self.held = True

    def release(self) -> None:
        if self.held:
            self.held = False
            self.slots.release()   # type: ignore[union-attr]


CURRENT_RUN: ContextVar[RunState | None] = ContextVar("sheath_current_run", default=None)


class AgentNode:
    """One agent in a conversation's tree: the top agent (depth 0) or a sub-agent."""

    def __init__(self, *, name: str, path: str, depth: int, creator: AgentNode | None = None,
                 agent: Any = None, description: str = "", bundles: tuple[str, ...] = (),
                 tool_objects: dict[str, Any] | None = None, can_spawn: bool = False,
                 limits: Limits | None = None, message_to: set[str] | None = None,
                 folder: str | None = None, declared: bool = False) -> None:
        self.name = name
        self.path = path
        self.depth = depth
        self.creator = creator
        self.agent = agent
        self.description = description
        self.bundles = bundles
        self.tool_objects: dict[str, Any] = dict(tool_objects or {})
        self.can_spawn = can_spawn
        self.limits = limits or Limits(timeout_s=None, max_round_trips=None,
                                       subtree_allowance=None, remaining_depth=0)
        self.message_to: set[str] = set(message_to or ())
        self.inbox: list[str] = []
        self.current_session: Any = None
        self.folder = folder
        self.declared = declared
        self._delegation: Delegation | None = None   # set when registered

    def __repr__(self) -> str:
        return f"AgentNode(path={self.path!r}, depth={self.depth})"

    @asynccontextmanager
    async def waiting(self) -> AsyncIterator[None]:
        """Wrap time this agent spends blocked on its own sub-agents.

        In a ``ConfinedSubagent`` run of this node, entering gives back the run's concurrency
        slot and pauses its ``subagent_task_timeout_s`` clock; leaving takes a slot again
        (queueing if none is free) and moves the deadline later by the time spent waiting. For
        the top agent, or outside a run of this node, it does nothing. Nested or parallel waits
        in one run release once and re-acquire when the last one leaves.

        On cancellation the slot is not taken back (the run is ending; its own teardown
        releases only a slot it holds), so a cancelled wait never leaks or over-releases one."""
        run = CURRENT_RUN.get()
        if run is None or run.node is not self:
            yield
            return
        loop = asyncio.get_running_loop()
        run._waiters += 1
        if run._waiters == 1:
            run._wait_began = loop.time()
            bound = run.bound
            if bound is not None and bound.when() is not None and not bound.expired():
                run._wait_deadline = bound.when()
                bound.reschedule(None)        # paused: waiting is not working time
            run.release()
        cancelled = False
        try:
            yield
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            run._waiters -= 1
            if run._waiters == 0:
                try:
                    if not cancelled and run.slots is not None and not run.held:
                        await run.acquire()
                finally:
                    deadline, run._wait_deadline = run._wait_deadline, None
                    if deadline is not None and run.bound is not None:
                        # Restored even on cancellation, so a run that survives it stays
                        # bounded; never in the past, so it cannot fire as a false timeout.
                        run.bound.reschedule(deadline + (loop.time() - run._wait_began))


class TaskRecord:
    """One delegated task: who started it, on which agent, and how it stands."""

    def __init__(self, *, id: int, owner: AgentNode, agent: AgentNode, description: str,
                 future: asyncio.Task | None = None, sub_session: Any = None) -> None:
        self.id = id
        self.owner = owner
        self.agent = agent
        self.description = description
        self.status: Literal["running", "completed", "failed"] = "running"
        self.result_text: str | None = None
        self.error_text: str | None = None
        self.future = future
        self.sub_session = sub_session
        # Set when a run ends (completed or failed); cleared when the owner reads the result.
        self.unread = False
        # Set when the host cancelled the run (Delegation.cancel_all): no notice, not unread.
        self.host_cancelled = False
        self._run: asyncio.Task | None = None    # the current run's future, kept once finalized
        self._read: asyncio.Task | None = None   # the run whose result the owner read


def _outcome(fut: asyncio.Task) -> Literal["completed", "failed"]:
    """How a finished future ended, as a task status."""
    return "failed" if fut.cancelled() or fut.exception() is not None else "completed"


def _finalize(record: TaskRecord) -> bool:
    """Take a finished future's outcome into the record (MAF's ``_finalize_task``).

    Returns True when it did (once per run: the future is dropped), else False."""
    fut = record.future
    if fut is None or not fut.done():
        return False
    if fut.cancelled():
        record.status, record.error_text = "failed", "Task was canceled."
    elif (exc := fut.exception()) is not None:
        record.status, record.error_text = "failed", str(exc)
    else:
        record.status, record.result_text = "completed", fut.result().text
    record.future = None
    return True


def _notice(record: TaskRecord, outcome: str) -> str:
    return (f"[task {record.id} ({record.agent.path}) {outcome} — read it with "
            "background_agents_get_task_results]")


def deliver(node: AgentNode, text: str) -> None:
    """Get ``text`` to ``node``'s model: into its next model call if a run of it is in
    progress (MAF's message injection drains the queue, mid-run included), else into its inbox,
    which is delivered when its next run starts."""
    session = node.current_session
    if session is None:
        node.inbox.append(text)
        return
    enqueue_messages(session, [Message(role="user", contents=[text],
                                       additional_properties={_DELIVERY_KEY: True})])


def _reclaim(node: AgentNode, session: Any) -> None:
    """Move deliveries still queued, undrained, on ``session`` back to ``node``'s inbox."""
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return
    with _MESSAGE_INJECTION_LOCK:
        queue = state.get(MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY)
        if not queue:
            return
        ours = [m for m in queue if m.additional_properties.get(_DELIVERY_KEY)]
        if not ours:
            return
        queue[:] = [m for m in queue if not m.additional_properties.get(_DELIVERY_KEY)]
    node.inbox.extend(m.text for m in ours)


def attach(node: AgentNode, session: Any) -> None:
    """A run of ``node`` starts on ``session``: deliveries go there, the inbox first.

    A previous session not detached (its run raised before ``after_run``) gives back what it
    still holds undelivered, so nothing is stranded on a session that will not run again."""
    previous = node.current_session
    if previous is not None and previous is not session:
        _reclaim(node, previous)
    node.current_session = session
    if node.inbox:
        pending, node.inbox[:] = list(node.inbox), []
        for text in pending:
            deliver(node, text)


def detach(node: AgentNode, session: Any) -> None:
    """A run of ``node`` on ``session`` ended: later deliveries wait in the inbox."""
    if node.current_session is session:
        _reclaim(node, session)
        node.current_session = None


def _log_abandoned(task: asyncio.Task) -> None:
    if not task.cancelled():
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _log.debug("abandoned sub-agent task raised: %s", type(exc).__name__)


class Delegation:
    """Every agent node and task of one ``Session``. Created lazily as ``session.delegation``."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._nodes: dict[str, AgentNode] = {}       # by name, case-folded
        self._tops: dict[str, AgentNode] = {}        # depth-0 nodes, by name
        self._records: list[TaskRecord] = []
        self._next_id = 1
        self.closed = False

    # --- nodes ------------------------------------------------------------------------------

    def top(self, name: str) -> AgentNode:
        """Get or create the depth-0 node called ``name``."""
        node = self._tops.get(name)
        if node is None:
            node = AgentNode(name=name, path=name, depth=0)
            node._delegation = self
            self._tops[name] = node
        return node

    def add_node(self, node: AgentNode) -> None:
        """Register a sub-agent node. Raises ``ValueError`` on a name taken ignoring case."""
        key = node.name.lower()
        if key in self._nodes:
            raise ValueError(f"agent name {node.name!r} is already used in this conversation "
                             f"(by {self._nodes[key].name!r})")
        node._delegation = self
        self._nodes[key] = node

    def node(self, name: str) -> AgentNode | None:
        return self._nodes.get(name.lower())

    def reachable(self, owner: AgentNode) -> list[AgentNode]:
        """The nodes ``owner`` may start tasks on, in creation order: those it created. For the
        top agent that is the host-declared roles plus the sub-agents it spawned; for a
        sub-agent, only the sub-agents it spawned (never a declared role or a sibling)."""
        return [n for n in self._nodes.values() if n.creator is owner]

    # --- tasks ------------------------------------------------------------------------------

    def _owned(self, owner: AgentNode) -> list[TaskRecord]:
        return [r for r in self._records if r.owner is owner]

    def _notify(self, record: TaskRecord, outcome: str | None = None) -> None:
        """Tell the owner its task ended (``outcome``, else the record's status), not what it
        returned: the result stays out of the owner's context until it reads it."""
        deliver(record.owner, _notice(record, outcome or record.status))

    def _watch(self, record: TaskRecord) -> None:
        """When ``record``'s current run ends, mark it unread and notify the owner, unless the
        host cancelled it or the owner already read its result.

        The record itself is still finalized lazily, as MAF's provider does, so the wait tool
        reports a run that ended since the last look as finished, in MAF's words."""
        fut = record.future
        if fut is None:
            return
        record._run = fut

        def _done(f: asyncio.Task) -> None:
            if record._run is not f or record._read is f or record.host_cancelled:
                return
            if record not in self._records:
                return   # cleared
            record.unread = True
            self._notify(record, _outcome(f))

        fut.add_done_callback(_done)

    def _refresh(self, owner: AgentNode) -> list[TaskRecord]:
        records = self._owned(owner)
        for r in records:
            if r.status == "running":
                _finalize(r)
        return records

    def _find(self, owner: AgentNode, task_id: int) -> TaskRecord | None:
        return next((r for r in self._refresh(owner) if r.id == task_id), None)

    def running_owned(self, owner: AgentNode) -> list[TaskRecord]:
        """``owner``'s tasks still running (refreshed first: a finished run is not counted)."""
        return [r for r in self._refresh(owner) if r.status == "running"]

    def unread_owned(self, owner: AgentNode) -> list[TaskRecord]:
        """``owner``'s finished tasks whose result it has not read since they finished."""
        return [r for r in self._refresh(owner) if r.status != "running" and r.unread]

    def start_task(self, owner: AgentNode, agent_name: str, input: str,
                   description: str) -> str:
        if self.closed:
            return "Error: Session is being released; cannot start a new background task."
        target = next((n for n in self.reachable(owner)
                       if n.name.lower() == agent_name.lower()), None)
        if target is None:
            available = ", ".join(n.name for n in self.reachable(owner))
            return (f"Error: No background agent found with name '{agent_name}'. "
                    f"Available agents: {available}")
        sub_session = target.agent.create_session()
        future = asyncio.create_task(target.agent.run(input, session=sub_session))
        record = TaskRecord(id=self._next_id, owner=owner, agent=target,
                            description=description, future=future, sub_session=sub_session)
        self._next_id += 1
        self._records.append(record)
        self._watch(record)
        return f"Background task {record.id} started on agent '{agent_name}'."

    async def wait_first(self, owner: AgentNode, task_ids: list[int], *,
                         timeout_s: float = DEFAULT_WAIT_TIMEOUT_S) -> str:
        if self.closed:
            return "Error: Session is being released; cannot wait for background tasks."
        if not task_ids:
            return "Error: No task IDs provided."
        by_id = {r.id: r for r in self._owned(owner)}
        waitable = [(tid, by_id[tid]) for tid in task_ids
                    if tid in by_id and by_id[tid].status == "running"
                    and by_id[tid].future is not None]
        if not waitable:
            done_already = next((r for r in self._refresh(owner)
                                 if r.id in task_ids and r.status != "running"), None)
            if done_already is not None:
                return (f"Task {done_already.id} is not running; "
                        f"current status: {done_already.status}.")
            return "Error: None of the specified task IDs correspond to running tasks."
        # Keep each future: while this waits, a parallel wait or a refresh may finalize the
        # record (future -> None), or continue_task may give it a new future.
        triples = [(tid, r, r.future) for tid, r in waitable]
        async with owner.waiting():   # a blocked sub-agent frees its slot and pauses its clock
            done, _ = await asyncio.wait({f for _, _, f in triples}, timeout=timeout_s,
                                         return_when=asyncio.FIRST_COMPLETED)
        if not done:
            return (f"No background task completed within {timeout_s} seconds. "
                    "The tasks are still running; call this tool again if you wish to "
                    "continue waiting.")
        completed_id, record, fut = next(t for t in triples if t[2] in done)
        if record.future is fut:
            _finalize(record)
        return f"Task {completed_id} finished with status: {_outcome(fut)}."

    def results(self, owner: AgentNode, task_id: int) -> str:
        record = self._find(owner, task_id)
        if record is None:
            return f"Error: No task found with ID {task_id}."
        if record.status != "running":
            record.unread, record._read = False, record._run
        if record.status == "completed":
            return record.result_text or "(no output)"
        if record.status == "failed":
            return f"Task failed: {record.error_text or 'Unknown error'}"
        return f"Task {task_id} is still running."

    def all_tasks(self, owner: AgentNode) -> str:
        records = self._refresh(owner)
        if not records:
            return "No tasks."
        return "\n".join(["Tasks:", *(_task_line(r) for r in records)])

    def continue_task(self, owner: AgentNode, task_id: int, text: str) -> str:
        if self.closed:
            return "Error: Session is being released; cannot continue a background task."
        record = self._find(owner, task_id)
        if record is None:
            return f"Error: No task found with ID {task_id}."
        if record.status == "running":
            return (f"Error: Task {task_id} is still running. "
                    "Wait for it to complete before continuing.")
        if record.agent.agent is None or self.node(record.agent.name) is not record.agent:
            return f"Error: Agent '{record.agent.name}' is no longer available."
        if record.sub_session is None:
            return f"Error: Session for task {task_id} is no longer available."
        record.future = asyncio.create_task(
            record.agent.agent.run(text, session=record.sub_session))
        record.status, record.result_text, record.error_text = "running", None, None
        record.unread = record.host_cancelled = False
        self._watch(record)
        return f"Task {task_id} continued with new input."

    def clear(self, owner: AgentNode, task_id: int) -> str:
        if self.closed:
            return "Error: Session is being released; cannot clear tasks."
        record = self._find(owner, task_id)
        if record is None:
            return f"Error: No task found with ID {task_id}."
        if record.status == "running":
            return (f"Error: Task {task_id} is still running. "
                    "Wait for it to complete before clearing.")
        record.unread = False
        self._records.remove(record)
        return f"Task {task_id} cleared."

    async def cancel_all(self) -> None:
        """Cancel every running task and registered sub-agent run, and wait for them, bounded
        by ``CANCEL_WAIT_S``. A task still pending after that is logged and left.

        A run ``ConfinedSubagent`` registers with the session on its first step; a task whose
        future is not recorded here (one started some other way) may not be registered yet. So
        after cancelling what it sees, this yields once and looks again, until a look finds
        nothing new (bounded by ``CANCEL_ROUNDS``). What it sees is cancelled before the yield,
        so none of it gets a step to work."""
        me = asyncio.current_task()
        tasks: list[asyncio.Task] = []
        for look in range(CANCEL_ROUNDS):
            seen = [r.future for r in self._records if r.future is not None]
            seen += list(self._session._subagent_tasks)
            new = [t for t in dict.fromkeys(seen)
                   if t is not me and not t.done() and t not in tasks]
            for record in self._records:
                if record.future is not None and record.future in new:
                    record.host_cancelled = True
            for task in new:
                task.cancel()
            tasks.extend(new)
            if look and not new:
                break   # a look after a yield found nothing new
            await asyncio.sleep(0)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=CANCEL_WAIT_S)
        for record in self._records:     # cancelled runs now read failed, not running
            if record.status == "running":
                _finalize(record)
        if pending:
            for task in pending:
                task.add_done_callback(_log_abandoned)
            _log.warning("%d sub-agent task(s) still running %.0fs after cancellation: %s",
                         len(pending), CANCEL_WAIT_S,
                         ", ".join(sorted(t.get_name() for t in pending)))


def _task_line(r: TaskRecord) -> str:
    return f"- Task {r.id} [{r.status}] ({r.agent.path}): {r.description}"


def _listing(nodes: list[AgentNode]) -> str:
    lines = ["Available background agents:"]
    for n in nodes:
        lines.append(f"- {n.path}: {n.description}" if n.description else f"- {n.path}")
    return "\n".join(lines)


class DelegationProvider(ContextProvider):
    """Give one agent node the six ``background_agents_*`` tools, MAF's names and strings."""

    def __init__(self, delegation: Delegation, node: AgentNode, *,
                 wait_timeout_s: int = DEFAULT_WAIT_TIMEOUT_S) -> None:
        super().__init__(_SOURCE_ID)
        if (isinstance(wait_timeout_s, bool) or not isinstance(wait_timeout_s, int)
                or wait_timeout_s <= 0):
            raise ValueError("wait_timeout_seconds must be a positive integer.")
        self._delegation = delegation
        self._node = node
        self._wait_timeout_s = wait_timeout_s

    async def before_run(self, *, agent: Any, session: Any, context: Any,
                         state: dict[str, Any]) -> None:
        d, node, wait_s = self._delegation, self._node, self._wait_timeout_s
        attach(node, session)

        @tool(name="background_agents_start_task", approval_mode="never_require")
        def background_agents_start_task(agent_name: str, input: str, description: str) -> str:
            """Start a background task on a named agent. Returns a confirmation with the task ID."""
            return d.start_task(node, agent_name, input, description)


        @tool(name="background_agents_wait_for_first_completion", approval_mode="never_require")
        async def background_agents_wait_for_first_completion(task_ids: list[int]) -> str:
            """Wait until the first of the specified tasks completes or the configured timeout expires.

            Returns the completed task's ID. On timeout, the tasks remain running and this tool
            can be called again to continue waiting.
            """
            return await d.wait_first(node, task_ids, timeout_s=wait_s)

        @tool(name="background_agents_get_task_results", approval_mode="never_require")
        def background_agents_get_task_results(task_id: int) -> str:
            """Get the text output of a background task by its ID."""
            return d.results(node, task_id)

        @tool(name="background_agents_get_all_tasks", approval_mode="never_require")
        def background_agents_get_all_tasks() -> str:
            """List all background tasks with their IDs, statuses, agent names, and descriptions."""
            return d.all_tasks(node)

        @tool(name="background_agents_continue_task", approval_mode="never_require")
        def background_agents_continue_task(task_id: int, text: str) -> str:
            """Send follow-up input to a completed or failed task to resume its work."""
            return d.continue_task(node, task_id, text)


        @tool(name="background_agents_clear_completed_task", approval_mode="never_require")
        def background_agents_clear_completed_task(task_id: int) -> str:
            """Remove a completed or failed task and release its session to free memory."""
            return d.clear(node, task_id)

        context.extend_instructions(
            self.source_id, [_PROVIDER_NOTE + "\n\n" + _listing(d.reachable(node))])
        sync_tools = [background_agents_start_task, background_agents_get_task_results,
                      background_agents_get_all_tasks, background_agents_continue_task,
                      background_agents_clear_completed_task]
        # On the event loop, not a worker thread: they read and change the shared task records
        # (and start/continue create tasks, which needs the running loop).
        for t in sync_tools:
            t._invoke_sync_on_event_loop = True
        context.extend_tools(self.source_id, [
            background_agents_start_task,
            background_agents_wait_for_first_completion,
            background_agents_get_task_results,
            background_agents_get_all_tasks,
            background_agents_continue_task,
            background_agents_clear_completed_task,
        ])
        records = d._refresh(node)
        if records:
            lines = ["### Current background tasks", *(_task_line(r) for r in records)]
            context.extend_messages(self.source_id,
                                    [Message(role="user", contents=["\n".join(lines)])])

    async def after_run(self, *, agent: Any, session: Any, context: Any,
                        state: dict[str, Any]) -> None:
        detach(self._node, session)


def tasks_running(node: AgentNode) -> Callable[..., bool]:
    """``should_continue`` for the parent loop: True while ``node`` has a task still running,
    or a finished one whose result it has not read since it finished (so a result that lands
    during the last pass is not dropped). Bounded by the loop's ``max_iterations``."""

    def _should_continue(**kwargs: Any) -> bool:
        d = node._delegation
        return d is not None and bool(d.running_owned(node) or d.unread_owned(node))

    return _should_continue


def tasks_running_message(node: AgentNode) -> Callable[..., str | None]:
    """``next_message`` for the parent loop: MAF's reminder listing ``node``'s running tasks,
    and a reminder to read its finished tasks' unread results."""

    def _next_message(**kwargs: Any) -> str | None:
        d = node._delegation
        running = d.running_owned(node) if d is not None else []
        unread = d.unread_owned(node) if d is not None else []
        parts = []
        if running:
            parts.append(
                f"You still have {len(running)} background task(s) running that must finish "
                f"before you can complete the work:\n{_brief(running)}\n\n"
                "Wait for these tasks to complete, retrieve their results, and incorporate them. "
                "Only stop once every background task has finished."
            )
        if unread:
            parts.append(
                f"You have {len(unread)} finished background task(s) whose results you have "
                f"not read:\n{_brief(unread)}\n\n"
                "Read each with background_agents_get_task_results and incorporate it before "
                "you finish."
            )
        return "\n\n".join(parts) or None

    return _next_message


def _brief(records: list[TaskRecord]) -> str:
    return "\n".join(f"- #{r.id} ({r.agent.path}): {r.description}" for r in records)
