# Sub-agents built by sheath — design

Date: 2026-11-04
Status: proposed. Written before implementation.
Issue: jspv/sheath#9.

## Problem

MAF's `create_harness_agent(background_agents=[...])` lets a model delegate work. The framework
adds a `BackgroundAgentsProvider` that gives the model six `background_agents_*` tools (start,
wait for first completion, get results, list, continue, clear), and each task runs concurrently
in its own `AgentSession`. sheath refuses `background_agents`, and should keep refusing it: an
agent the caller supplies is arbitrary, and its tools can sit outside the sandbox, `safe_path`
and the egress guard. As a result, a sheath agent cannot delegate at all, even to an agent that
would be confined exactly as it is.

## Goal

Let a host declare named sub-agents that **sheath builds itself**, over the parent's own
`Session`. A sub-agent can then do nothing the parent could not.

Non-goals:
- caller-supplied agents;
- nesting (sub-agents of sub-agents);
- sub-agent state surviving a process restart (MAF marks lost tasks `lost`).

## Facts this rests on (agent-framework-core 1.22.0, checked by a spike)

- `start_task` calls `agent.create_session()` and then
  `asyncio.create_task(agent.run(input, session=sub_session))`. The result the parent reads is
  `response.text`. `continue_task` calls `agent.run` again on the same sub-session.
- **A wrapper object works.** Any object with `name`, `id`, `description`, `create_session`,
  `get_session` and an async `run` can intercept the run and replace the returned text. The
  spike's wrapped results reached the parent.
- **A contextvar set inside the wrapper's `run` reaches that sub-agent's tool calls and nothing
  else.** It is set inside the task the provider created, so the parent's context never sees it.
- **Tool errors.** A sub-agent tool that raises gives the model `"Error: Function failed."`. The
  `function_result` content's `exception` carries the message, for example
  `path escapes root: '../out.txt'`.
- **Cleanup.** `BackgroundAgentsProvider.release_session(session)` cancels running tasks (bounded
  at 30 s). Nothing in the framework calls it.
- **Wait timeout.** The wait tool's timeout is `wait_timeout_seconds` (default 300). On expiry it
  returns without cancelling anything.
- **Experimental.** The provider class is marked `ExperimentalFeature.HARNESS`.
- **Tool-loop limits are per client** (see #8, `function_invocation_configuration`).
- **Re-running the parent.** `create_harness_agent(loop_should_continue=..., loop_next_message=...,
  loop_max_iterations=...)` wraps the agent in `AgentLoopMiddleware`, which re-runs it while the
  predicate returns true.
  - `agent_framework.background_tasks_running()` is the stock predicate: true while the agent's
    `BackgroundAgentsProvider` has a task recorded as running.
    `agent_framework.background_tasks_running_message` is the matching nudge.
  - The cap defaults to 10 runs.
  - Without an `AgentSession` the predicate is false, so a bare `run()` does not loop.
  - Streaming runs yield every pass's updates, plus each nudge as a `role="user"` update.
    `return_final_only` affects only non-streaming runs.
- **AG-UI** (`agent-framework-ag-ui`, captured through `agui_event_stream` with a stub parent):
  - the parent's `background_agents_*` calls stream as ordinary `TOOL_CALL_*` events;
  - a sub-agent's own calls and text never enter the stream, because MAF runs it with
    non-streaming `run()` in a background task;
  - the MAF session id is the `threadId`, the same as the sheath conversation id;
  - `merge_status` flushed a status event emitted at 0.18 s only at 0.78 s, when the parent's
    wait tool returned;
  - the loop's nudge was folded into the assistant's text and into `MESSAGES_SNAPSHOT`;
  - an `AgentMiddleware` placed before the loop, which drops `role="user"` updates from the
    `ResponseStream` and keeps the inner final response, removed the nudge and left the model's
    text intact.

## Design

### Public surface

```python
@dataclass(frozen=True, kw_only=True)
class SubagentSpec:
    description: str                         # what it is for; shown to the parent model
    instructions: str | None = None          # its domain prompt (agent_instructions)
    bundles: tuple[str, ...] | None = None   # None = the parent's selection
    tools: tuple | list = ()                 # callables, ToolFactory, MCP tools
    model: str | None = None                 # model for a client sheath builds; None = config.model
    client: Any | None = None                # a ready client of its own, instead of model
    max_round_trips: int | None = None       # per-task tool-loop budget
```

A sub-agent may run on a different model, or a different provider, from its parent. sheath
expects that rather than treating it as an exception.

- `Session.create_agent(..., subagents: Mapping[str, SubagentSpec] | None = None,
  client_factory: Callable[[str], Any] | None = None)`. The factory takes a model name and
  returns a new client for it.
- `Sheath(..., subagents=..., client_factory=...)`. With no `client_factory`:
  - if you injected no `client`, the facade uses its built-in one,
    `lambda model: OpenAIChatClient(model=model, env_file_path=".env")`;
  - if you did inject one, there is no factory, because sheath cannot clone a client it did not
    build.
- `aopen(..., subagents=...)`, `asolve(...)`, `Sheath.solve(...)` and module-level `solve(...)`
  take `subagents` too. A per-call value that is not `None` replaces the `Sheath` default whole,
  the same precedence as `agent_instructions`. `{}` means no sub-agents.
- **How the parent model picks a model today.** MAF's `start_task` takes only an agent name, so
  the host declares one sub-agent per role and model, for example `researcher` on a strong model
  and `skimmer` on a cheap one. The `description` field tells the parent which to use for what.
  - Out of scope here: a sheath-owned start tool taking a model from a host allowlist. The
    factory's `(model) -> client` shape is what that would need, so nothing here would change.
- `SubagentSpec` is exported from `sheath`.
- Two new `SheathConfig` fields:
  - `subagent_max_concurrent: int = 2`: tasks allowed to run at once, per session;
  - `subagent_task_timeout_s: float | None = 600.0`: per-task wall-clock bound, measured from
    when the task gets a slot, so waiting in the queue does not count against it.
- MAF's own wait timeout stays a MAF option. If a caller passes
  `background_agents_wait_timeout_seconds` in `agent_kwargs`, sheath forwards it to the
  provider.

### Validation (all at `create_agent`, before anything connects)

| Condition | Error |
|---|---|
| `background_agents` among the kwargs | `TypeError` (unchanged) |
| `SubagentSpec(subagents=...)` | `TypeError` (the field does not exist) |
| a value in `subagents` that is not a `SubagentSpec` | `TypeError` |
| name not matching `^[A-Za-z][A-Za-z0-9_-]{0,63}$`, or two names equal case-insensitively | `ValueError` |
| a spec bundle outside the parent's selected bundles | `ValueError` |
| `max_round_trips < 1` | `ValueError` |
| both `model` and `client` set | `ValueError` |
| `spec.client` is the parent's client object, or the same object as another spec's client | `ValueError` |
| no `spec.client` and no `client_factory` | `ValueError` naming both ways to fix it |
| `subagent_max_concurrent < 1`, or `subagent_task_timeout_s <= 0` | `ValueError` |

**Why bundles must be a subset.** It is what makes "nothing the parent could not" hold for the
built-in tools. A spec's extra `tools` are the host's own choice, the same trust as the parent's
`tools`.

### Building a sub-agent

For each spec, `create_agent` makes a recursive call to its own build path on the **same
`Session`**. The sub-agent therefore gets the same root, `HandleStore`, sandbox, egress policy,
input-tree context provider, compaction defaults and refused-kwargs check. It is built with:
- `name=<name>` and `subagents=None`;
- `agent_instructions=spec.instructions`;
- the core and bundle manual, plus a new `SUBAGENT_INSTRUCTIONS` fragment: its final message is
  a report to the delegating agent; files and handles are shared, so it should name them rather
  than paste them;
- `tools=expand_tools(spec.tools)`. A `ToolFactory` is built once per sub-agent per
  `create_agent` call, so once per conversation. MCP tools go through `_attach_mcp`, so they are
  closed when the session closes. Their status events carry the sub-agent's name.

**Client resolution.** Every sub-agent gets a client of its own, never the parent's:
1. `spec.client`, if given; it must be distinct from the parent's and every other spec's;
2. otherwise `client_factory(spec.model or config.model)`, called once per sub-agent per
   `create_agent`, so once per conversation.

**What that buys.** Because no client is shared:
- the `SheathConfig` tool-loop limits and a spec's `max_round_trips` apply to that client
  alone, through `_apply_tool_loop_limits`, and never change the parent's;
- concurrent sub-agents never share a client's connection state.

### The provider and the wrapper

- sheath constructs `BackgroundAgentsProvider` itself and adds it to `context_providers`. It
  does not pass `background_agents=` to MAF, so it holds the provider for release.
- `instructions` is sheath's note followed by MAF's agent listing:
  `"Sub-agents work in this workspace under the same limits as you. Delegate a self-contained
  side task with background_agents_start_task, wait for every task you start before you finish,
  and open a long result with inspect_handle.\n\n{background_agents}"`.
- Each sub-agent is wrapped in `ConfinedSubagent`. `name`, `id`, `description`,
  `create_session` and `get_session` delegate to the built agent. `run(messages, *, stream=False,
  session=None, **kw)` does the following:
  1. refuses `stream=True` (`ValueError`; the provider never streams);
  2. registers the current task with the session, so close can cancel it, queued or running;
  3. acquires the session's semaphore, created lazily with `subagent_max_concurrent` slots;
  4. binds the agent tag (`status.bind_agent(name)`) and emits `StatusEvent(tool="sheath",
     message="started", agent=name)`;
  5. runs the inner agent under `asyncio.wait_for(..., subagent_task_timeout_s)`. A timeout
     raises `TimeoutError("sub-agent 'x' ran past subagent_task_timeout_s (600s)")`, which MAF
     records as the task's `error_text`;
  6. spills the text with `_maybe_spill(session, name, text, source="subagent:<name>")`. A
     result over `spill_threshold_bytes` becomes a handle, and the parent sees the summary as
     JSON text. A result over `max_spill_bytes` fails the task with `SpillLimitExceeded`;
  7. emits `finished`, or `failed: <ExceptionType>` (the type only; see Status tagging), and returns a fresh `AgentResponse` holding that
     text.

### The parent waits

A parent turn does not end while a task it started is still running.

**Enforced, not just instructed.** When a parent has sub-agents, `create_agent` builds the loop
middleware itself and puts it first in `middleware`, ahead of the caller's:

```python
AgentLoopMiddleware(background_tasks_running(),
                    next_message=background_tasks_running_message,
                    return_final_only=True, inject_progress=False,
                    max_iterations=<loop_max_iterations from agent_kwargs, else MAF's 10>)
```

**Why not `loop_should_continue=`.** With that route, the loop returns every pass joined together
("finished early", a "Progress so far" block, then the real answer), and `create_harness_agent`
does not pass `return_final_only` through. A spike on 1.22.0 confirmed that the explicit
middleware returns only the final answer.

- **The nudge** is MAF's stock `background_tasks_running_message`. It names the tasks still
  running and tells the parent to wait for them and read their results.
- **The caller's own loop.** If the caller passed `loop_should_continue` in `agent_kwargs`,
  sheath adds no loop of its own; the caller has taken over that decision.
- **Placement.** The loop sits just inside MAF's tool-approval middleware rather than outermost.
  That only matters if a caller re-enables approval, which sheath turns off.

**What bounds the wait.**
- Each task's own `subagent_task_timeout_s` ends it. A task that times out is `failed`, not
  `running`, so the loop stops.
- The parent's own limits still apply. Each `wait_for_first_completion` call is one round-trip
  and blocks for up to MAF's `wait_timeout_seconds` (300 s by default). A 600 s task therefore
  costs about two round-trips, well inside a 40-round-trip budget. A host with long sub-agent
  tasks should set generous parent limits; the README says so.

**Tasks belong to the turn.** If the parent turn still ends with a task running (the re-run cap
reached, an error, or a client disconnecting from an AG-UI stream), `Conversation.aask` and
`astream_agui` cancel the leftover tasks in a `finally` before releasing the turn lock.
- A sub-agent therefore never runs between turns, so the idle sweep needs no special case.
- `continue_task` still works on a task that completed in an earlier turn: its sub-session
  stays until cleared.

### Status tagging

- `StatusEvent` gains `agent: str | None = None`.
- `StatusBus.emit` fills it from a new `_current_agent` contextvar when the event has none. That
  covers `report_progress` from built-in and host tools, including sync tools run through
  `asyncio.to_thread`, which copies the context.
- MCP notifications arrive on the MCP client's own task, so `install_status_wrappers` and
  `_attach_mcp` take `agent=` and stamp it directly.
- The AG-UI `sheath.status` value gains `"agent"`. The CLI's verbose line becomes
  `→ [<agent>] <tool>: ...` when the field is set.
- `bind_agent` restores the previous value with `set()`, not `reset(token)`, for the same
  reason as `bind_bus`.
- **Per-tool activity.** Each sub-agent gets a sheath function middleware, `_ToolActivity`, that
  emits an event around every tool call:
  - `StatusEvent(tool=<tool>, message="running")` before the call;
  - `"done"` after it, or `"failed: <ExceptionType>"` if it raises, then re-raises.

  The tag fills in `agent`. Most built-in tools never call `report_progress`, so without this a
  host would see only "started", a silence and "finished". The exception message is left out on
  purpose: it can echo untrusted content, and the host already gets the full error through the
  task's result.
- The parent gets no `_ToolActivity`. Its tool calls already stream as AG-UI `TOOL_CALL_*`
  events and as MAF response updates.

### AG-UI

What a frontend sees:
- **Delegation as tool calls.** `background_agents_start_task` (with `agent_name`, `input` and
  `description`), `wait_for_first_completion` and `get_task_results` stream as `TOOL_CALL_*`
  events. The result carries the report, or the handle summary when it was spilled. Frontends
  render these with their usual backend-tool renderers.
- **Sub-agent activity as `sheath.status` custom events.** These are the `started`, per-tool
  and `finished`/`failed` events, each value carrying `"agent"`.

Two sheath changes make that work:

1. **Live status.** `merge_status` waits on the next source event and the status queue at the
   same time:
   - it keeps one pending `__anext__` task on the source;
   - it yields queued status the moment it lands, including while the parent sits in
     `wait_for_first_completion`;
   - it cancels the pending task on early exit.

   Ordering is unchanged: within each stream, events keep their order.
2. **No nudges in the chat.** `create_agent` puts `_DropLoopNudges`, an `AgentMiddleware`, ahead
   of its loop middleware, whenever it adds that loop. On a streaming run it:
   - wraps `context.result` in a new `ResponseStream` that skips updates whose role is `user`;
   - has a finalizer that returns the inner stream's final response.

   Non-streaming runs pass through. The parent's own premature text, for example "finished
   early", stays visible: it is real model output.

**Unverified, pinned by a test.** It is not yet known whether `continue_task` works on a task
from an earlier AG-UI request. The wrapper may rebuild session state from the replayed messages
each request, which would make the answer "No task found". The test records whichever is true,
and the README states it.

### Lifecycle

- `Session.subagents_running -> int` counts registered tasks, queued or running.
- **`Session.aclose`, first step:** cancel every registered task and await them, bounded by
  `_SUBAGENT_CANCEL_WAIT_S = 30.0`. After that it closes MCP servers and applies `cleanup`. A
  sub-agent therefore never outlives its MCP servers or its root, whoever built the agent.
- `Session.release_subagents(agent_session)` calls `release_session` on each provider this
  session built.
- **`Conversation.aclose`:** after the turn-lock wait, call `release_subagents(agent_session)`,
  then `session.__aexit__`. Then reap the root in a worker thread while holding
  `session.io_lock`, so a sync file tool already in flight on a worker thread finishes before
  the directory goes. `Session.aclose`'s `cleanup` removal uses the same helper.
- `Session.cancel_subagents()` cancels every registered task and awaits it, bounded by
  `_SUBAGENT_CANCEL_WAIT_S`. `aclose` and the turn-end `finally` both use it.
- `Session.subagent_names -> tuple[str, ...]` records what `create_agent` built. Hosts and tests
  read it.

## Confinement argument

A sub-agent's capability set is:
- the built-in tools of a subset of the parent's bundles, bound to the parent's `Session`;
- plus tools the host named in its spec.

Every built-in tool goes through the same `safe_path`, sandbox and egress code as the parent's,
because it is the same `Session` object. The refused kwargs (`shell_executor`,
`file_access_store`, `background_agents`) are checked on the sub-agent's build too. Since
`subagents=None` there, the sub-agent has no delegation tools.

The residual risk is MAF's own: a sub-agent's report is model text from content that may be
untrusted. It reaches the parent the way any tool result does, as data, and it is spilled when
large. The README says this.

Concurrency adds no new path:
- `run_python` is already serialized per workspace by the sandbox lock;
- file tools take `file_io`;
- `HandleStore` is already safe under concurrent `put`.

## Testing

- **Acceptance tests from the issue.** These use `StubChatClient`. Its script now accepts a
  callable step `(messages) -> Content`, so a stub parent can echo the results it was given.
- **Validation table:** one test per row.
- **Lifecycle:**
  - closing a conversation cancels a sleeping sub-agent before its root is removed;
  - `release_session` is called;
  - a parent scripted to finish without waiting is re-run, and its final text holds the result;
  - a task still running when the turn ends is cancelled at turn end;
  - the wall-clock bound fails a task with the message above.
- **Status:** `agent` is set on events from the sub-agent's tools and its MCP server, and is
  `None` on the parent's. Every sub-agent tool call yields `running`, then `done` or `failed`.
- **AG-UI:**
  - a status event emitted during a blocked parent tool is yielded before that tool's result;
  - no nudge text appears in `TEXT_MESSAGE_CONTENT` or in `MESSAGES_SNAPSHOT`;
  - the `continue_task`-across-requests test.
- **Facade precedence and client factory.**
- **One opt-in live test** (`SHEATH_LIVE=1`): a real parent delegates one fetch-and-count task.
