# Runtime sub-agents (`spawn_subagent`) and agent messaging (`send_message`) — design

Date: 2026-11-05
Status: proposed. Written before implementation.
Builds on: `2026-11-04-subagents-design.md` (issue #9). That design's confinement, workspace,
spill, status, lifecycle and AG-UI rules all still apply unless this document says otherwise.

## Intent

Let an agent decide, mid-run, that it needs help, and create sub-agents itself. Example: a cyber
investigator pulls data from a SIEM, sees an alert, and decides that four leads should be chased
in parallel. It creates four sub-agents on the spot, each with instructions it writes from what
it just saw, the model that suits the task, and the tools that lead needs. When the results come
back, it decides the next step: more sub-agents, continuing one of them, or finishing.

The owner's reasons for sub-agents, which every decision below serves:
- **Shared workspace.** The parent tells a sub-agent to read or write files in the workspace; the
  handoff goes both ways through files, not only through the reply.
- **Model choice.** Each task can run on a more or less capable model than the parent's.
- **A clean context for the sub-agent.** It starts from the instructions the parent gives it, not
  the parent's history.
- **A clean context for the parent.** The parent sends a sub-agent to do the digging and gets back
  only the result.

Today (after #9) sub-agents can only be declared by the host in code, with fixed roles. The
roles a real investigation needs are not known ahead of time.

## Goals

1. A `spawn_subagent` tool the agent calls to create a sub-agent and start its first task in one
   step. The agent chooses the name, the instructions, the task, the model, the tools, and whether
   the sub-agent may itself create sub-agents.
2. The spawned sub-agent is confined exactly like a #9 sub-agent: same `Session`, workspace,
   `safe_path`, sandbox, egress, spill, status tagging, lifecycle. It can do nothing its creator
   could not.
3. `spawn_subagent` is a tool like any other. A creator that hands it down (by naming it in the
   new sub-agent's `tools`) gives that sub-agent the ability to create sub-agents; it can hand down
   only what it has, within the host's depth limit (owner's decision: no separate delegation
   flag).
4. The host bounds everything in code: which models may be used, how many sub-agents may be
   created in a conversation, how deep delegation may go. Requests beyond those limits are
   refused, as a tool error the agent can read and adapt to.
5. The existing delegation tools (start, wait, get results, continue, list, clear) work on spawned
   sub-agents exactly as on declared ones. Declared sub-agents (#9) keep working alongside.
6. Agents can message each other while they work (`send_message`), anywhere the creating agent
   permits, siblings included. Like delegation, messaging is a capability the creator grants, and
   the creator's instructions tell a sub-agent whom to message and when. The creator can grant or
   withdraw messaging permissions at any time, not only when it creates a sub-agent.
7. Delegating never blocks the creator. While its sub-agents run, an agent keeps working with its
   own tools, and learns of each finished task at its next step, without having to call the wait
   tool. Waiting is something it chooses to do when it has nothing else to do.

## Non-goals

- The agent defining new tools or code-level capabilities. It can only hand down tools it has.
- Sub-agents that survive the conversation, a process restart, or move between conversations.
- Letting the agent raise any host limit.

## Anti-goals (failure even if it technically works)

- Any path by which a sub-agent ends up with a tool, bundle, model or delegation right its creator
  did not have, or that the host did not allow.
- Unbounded creation: a prompt-injected alert telling the parent to spin up fifty sub-agents on
  the most expensive model must hit the host's caps and be refused.
- A silent fallback: a refused model or tool is never quietly replaced with another.
- A sub-agent (or its sub-agents) still running after its creator's task, the turn, or the
  workspace is gone.

## Host configuration

```python
@dataclass(frozen=True, kw_only=True)
class SpawnPolicy:
    models: tuple[str, ...]          # allowlist; required, non-empty
    default_model: str | None = None # used when the agent names none; None = models[0]
    max_subagents: int = 8           # spawned per conversation, at every depth, ever
    max_depth: int = 3               # backstop on nesting: depth of the deepest sub-agent
    max_messages: int = 200          # send_message calls per conversation
    max_message_chars: int = 4000    # per message
```

- `Sheath(..., spawn=SpawnPolicy(...))`, with per-call `aopen(..., spawn=...)` and
  `asolve(..., spawn=...)` overrides, same precedence as `subagents` (a value that is not `None`
  replaces the default whole). `Session.create_agent(..., spawn=...)` underneath.
- No policy means no `spawn_subagent` tool: spawning is off unless the host turns it on.
- Clients come from `client_factory(model)`, as for declared sub-agents. A policy without a
  usable factory raises `ValueError` at `create_agent`. Every spawned sub-agent gets a client of
  its own (never the parent's, never another sub-agent's), reusing #9's identity checks.
- Validation at `create_agent`: `models` non-empty, each a non-empty string; `default_model` in
  `models`; `max_subagents >= 1`; `max_depth >= 1`. Errors are `ValueError`.
- Declared sub-agents gain `SubagentSpec.can_spawn: bool = False`, so a host can hand
  `spawn_subagent` to a declared role too (only meaningful with a policy).

## The tool

`spawn_subagent(name, instructions, task, model=None, tools=None, bundles=None, message_to=None, description="", timeout_s=None, max_round_trips=None, max_subagents=None, max_depth=None)`

(`message_to` is described under "Messaging between agents"; refused when it names an agent the
creator may not message.)

| Argument | Meaning | Refused (tool error, nothing created) when |
|---|---|---|
| `name` | new sub-agent's name | not matching #9's pattern `^[A-Za-z][A-Za-z0-9_-]{0,63}$`, or equal (case-insensitively) to any sub-agent name already in this conversation, declared or spawned, at any depth |
| `instructions` | the sub-agent's whole role prompt | empty |
| `task` | the first task's input | empty |
| `model` | model name | not in `models` |
| `tools` | names of the creator's own tools to hand down: host and MCP tools, and `spawn_subagent` itself | any name the creator does not have (the error lists the names it does have); `spawn_subagent` when the new sub-agent's depth would exceed `max_depth` |
| `bundles` | built-in bundles to hand down; default: all the creator has | any bundle the creator does not have |
| `description` | one line shown in task listings and status | — |
| `timeout_s` | wall-clock bound on each of the sub-agent's tasks (its own working time, as in "Concurrency and time under nesting") | not positive, or above the creator's own effective bound |
| `max_round_trips` | model round-trips per task, applied to the sub-agent's own client | below 1, or above the creator's own effective budget |
| `max_subagents` | how many sub-agents this sub-agent and everything beneath it may create, in total | given without handing down `spawn_subagent`; below 1; or above the creator's own remaining allowance |
| `max_depth` | how many further levels of nesting may exist beneath this sub-agent | given without handing down `spawn_subagent`; below 1; or deeper than the creator's own remaining depth |

- The total cap is checked last: if `max_subagents` sub-agents were already spawned in this
  conversation (any depth), the call is refused with how many were used of how many allowed.
  A refused call does not count.
- On success it returns the same confirmation shape as `start_task` (the task id and the
  sub-agent's name), so the agent continues with the existing wait / get results / continue
  tools. The spawned sub-agent can also be given later tasks with `start_task` by name.
- Default tools: none of the creator's host or MCP tools unless named. Built-in bundles default
  to the creator's own.

### Per-sub-agent limits (owner's decision: the creator may tighten, never loosen)

Every sub-agent has an *effective* value for each of the four limits. The top agent's are the
host's: `subagent_task_timeout_s`, the client's round-trip setting (`tool_loop_max_iterations`,
or MAF's 40), `SpawnPolicy.max_subagents`, `SpawnPolicy.max_depth`. A sub-agent's effective value
is the one its creator gave it, or, when omitted, the creator's own effective value (for depth:
the creator's remaining depth minus one). A value above the creator's effective value is refused,
so limits only tighten down the tree.

- `timeout_s` and `max_round_trips` apply to each of the sub-agent's tasks, including later
  `start_task` and `continue_task` calls on it.
- `max_subagents` is an allowance for the sub-agent's whole subtree, not a reservation: spawns
  anywhere beneath it count against it *and* against every allowance above it, up to the
  conversation-wide `SpawnPolicy.max_subagents`. The first spawn that would exceed any of them is
  refused, naming which limit. This stops the first sub-agent given spawning from using up the
  whole tree's budget.
- `max_depth` counts levels beneath the sub-agent: `max_depth=1` means its own sub-agents may not
  spawn. It is capped by the host's `SpawnPolicy.max_depth` as well.
- Effective limits are fixed when the sub-agent is created (my call), and shown in its tool
  descriptions (remaining allowance, depth) so it can plan.
- The tool's description lists the allowed models, the default model, and how many sub-agents
  remain in the budget, refreshed every turn, so the agent can plan without trial and error.

## Working while sub-agents run (owner's decision)

Starting or spawning a task returns at once, and the creator carries on with its own work. The
wait tool is optional: it is for when the creator has nothing else to do.

- **Completion notices** (my call on the mechanism): when a task an agent started finishes, fails
  or times out, a short notice is injected into that agent's next model call, through the same
  message-injection path as `send_message`: `[task 3 (lead-2) completed — read it with
  background_agents_get_task_results]`. The result itself is not injected, so the creator's
  context stays clean until it chooses to read it.
- The parent-wait loop is unchanged in purpose: an agent cannot *finish* its turn while tasks it
  started are still running. It can do anything else in the meantime.
- An agent that keeps working holds its concurrency slot (if it is a sub-agent) and its clock
  keeps running; only time spent waiting is excluded (next section).

## Reach: what each agent's delegation tools can see (my call)

Each agent's delegation tools reach only the sub-agents it spawned itself and the tasks it
started; the top agent also reaches the host-declared roles. A sub-agent cannot start, continue,
list, read or clear tasks of a declared role, a sibling, or a sibling's sub-agents. Reason:
declared roles carry tools the host chose for them, and reaching them would let a sub-agent use
those tools by proxy, breaking the subset rule.

- Tasks are keyed by their owner (the agent that started them); each agent's tools show its own.
- A sub-agent's own sub-agents and their tasks belong to it and persist across its tasks: if the
  parent later continues `lead-2`, `lead-2` still sees its earlier sub-agents and their results.
- The parent-wait loop counts only the waiting agent's own running tasks.

## Concurrency and time under nesting (owner's decision)

- An agent that is blocked waiting on its own sub-agents (inside the wait tool, or re-run by the
  parent-wait loop while only waiting) gives up its concurrency slot and re-acquires it when it
  resumes. `subagent_max_concurrent` therefore means "at most N agents actively working at once";
  waiting agents do not count.
- Time spent waiting on its own sub-agents does not count against an agent's
  `subagent_task_timeout_s`; the bound measures the agent's own working time.
- Total time stays bounded: every sub-agent has its own bound, depth and count are capped, and
  the parent-wait loop's re-runs are capped (`loop_max_iterations`).

## Depth, naming and nesting

- The top agent is depth 0; its sub-agents depth 1; theirs depth 2. A sub-agent may spawn only if
  its creator handed it the `spawn_subagent` tool. Handing it down is refused when the new
  sub-agent would sit deeper than `max_depth` (default 3: the top agent, its sub-agents, theirs,
  and one more level; a host that wants no nesting sets 1). The total cap bounds the tree
  regardless of depth.
- Every limit applies to the whole tree: the model allowlist, the concurrency cap, the per-task
  wall-clock bound, and the total cap (one counter per conversation).
- Status events from nested sub-agents carry the path as their `agent`: `lead-2/whois`. Names
  must still be unique in the whole conversation, so the last segment alone also identifies one.
- A sub-agent that delegates waits for its own sub-agents before reporting, exactly as the top
  agent does now.
- **Cascade.** When a sub-agent's task ends for any reason (finished, failed, timed out,
  cancelled), every task it started that is still running is cancelled. Nothing runs under a
  creator whose task is gone. Turn end and workspace close still cancel everything, as in #9.

## Messaging between agents (`send_message`)

**What it is.** A tool `send_message(to, text)`. The recipient sees the message at its next model
call, labelled with the sender's path (`[message from lead-1] ...`). It is fire-and-forget: the
sender carries on; a reply, if any, comes back the same way. Uses: the parent redirects a running
sub-agent; one lead tips off another; a sub-agent tells its creator something urgent before it
finishes.

**Delivery** (verified: MAF 1.22.0's always-on `MessageInjectionMiddleware` drains messages queued
on an `AgentSession` into that session's next model call, including while a run is in progress).
- A message to an agent with a running task is queued on that task's session and seen at its next
  model call, between tool rounds, without stopping it.
- A message to an agent with no running task is held and delivered at the start of its next task
  (a later `start_task` or `continue_task`). Held messages are dropped when the conversation
  closes.
- A message to an agent blocked in the wait tool wakes the wait early (the tool returns saying a
  message arrived), so a waiting parent can react without waiting out the timeout.

**Permission: anywhere the creator permits, siblings included, changeable at any time** (owner's
decisions).
- Every agent has `send_message` whenever the host's policy is on; what it may do with it is its
  permission list, which starts empty (only its creator, see below) and can change while it runs.
  An agent with nothing permitted gets a refusal explaining that it may not message anyone yet.
- The creator sets a sub-agent's list at spawn time, `spawn_subagent(..., message_to=[...])`, and
  changes it later with `set_message_permissions(agent, allow=[...], deny=[...])`, for example when
  it decides mid-investigation that `lead-1` should tip off `lead-3`. A change takes effect for the
  next `send_message` call, including in a task that is already running. Withdrawing a permission
  does not recall messages already sent. Declared roles get an initial `SubagentSpec.message_to`
  set by the host, and the top agent can change it like any other.
- `set_message_permissions` reaches only the agents the caller may reach (its own sub-agents; the
  top agent also declared roles), as with the delegation tools.
- A sub-agent may always message its creator, so it can report something urgent or answer a
  question without a grant (my call: replying to whoever delegated is the common case and widens
  nothing). The creator can withdraw that too with `deny`.
- Subset rule: a creator can grant only names it may message itself. The top agent may message
  every agent in the conversation. Names may refer to agents that do not exist yet (a sibling
  spawned later); permission is checked when a message is sent, and a message to a name that
  does not exist or is not permitted is refused as a tool error.
- Messaging grants no tools and no reach: it only delivers text.

**Safety.**
- Messages are model-written text that may carry injected content. They arrive labelled with the
  sender and framed as a message from another agent, to be treated as information, never as host
  instructions; the sub-agent note says so.
- Per-message size cap (`SpawnPolicy.max_message_chars`, default 4000; longer text is refused,
  with a hint to write a workspace file and send its path).
- A per-conversation message cap (`SpawnPolicy.max_messages`, default 200) stops two agents
  ping-ponging forever; beyond it, `send_message` is refused.
- Status events: each delivered message emits `StatusEvent(tool="sheath", message="message to
  <recipient>", agent=<sender path>)` so a host can show the traffic. The text itself is not put
  in status.

**Messaging is phase 2.** It is built after spawning works, as its own reviewed steps.

## Shared workspace: many agents, no clobbering

Several agents now read and write the same workspace at once. What exists today, and what this
feature adds:

**Already safe (verified in the code):**
- File tools and sandbox runs take one workspace lock, so two writes never interleave and no file
  is ever half-written by two agents at once.
- Handle ids the store assigns (`h1`, `h2`, ...) are allocated under a lock and never collide.
  A handle id an agent chooses itself (`save("summary", ...)` in `run_python`) that already exists
  is refused: handles are immutable, so a second agent cannot overwrite the first one's handle.

**Added: each sub-agent has its own folder.** Every sub-agent gets a working folder
`agents/<path>/` (for example `agents/lead-2/` or `agents/lead-2/whois/`), created when it is
spawned. Its instructions say to put its own files there unless its creator told it to use
another path, and to prefix handle ids it picks with its name. New files from different agents
then do not collide by name. The folder is a convention, not a jail: an agent can still read and
write anywhere in the workspace, which is the point of a shared space (the parent may tell a
sub-agent to read `evidence/alert.json` and write `findings/lead-2.md`).

**Added: no silent overwrites between agents.** The real risk is a lost update: `lead-1` writes
`findings/summary.md`, then `lead-3` overwrites it without having seen it. sheath tracks, per
file, which agent last wrote it through the file tools and the content's hash, and, per agent,
the hash of each file as that agent last read or wrote it.
- `write_file` or `edit_file` on an existing file whose last write came from a *different* agent
  is refused unless the caller has read the current content since that write. The error names the
  other agent and says to read the file first or write elsewhere.
- An agent rewriting its own file, or any file no other agent has written, behaves exactly as
  today. A single-agent run never sees a difference.
- `edit_file` already replaces atomically; `write_file` becomes atomic too (temporary file plus
  rename), so a concurrent reader never sees a partial file.

**Limit (documented):** files written by code inside `run_python` are not attributed to an agent
(the sandbox cannot tell sheath which agent's code wrote what beyond the run itself), so the
overwrite guard covers the file tools, not arbitrary code. Code still runs one script at a time
per workspace.

## What a sub-agent sees

- Its instructions are the parent-written `instructions` plus sheath's operating manual for its
  bundles, plus the sub-agent note. The note is strengthened for this feature: the final message
  is a concise result for the agent that delegated; bulky material goes in workspace files or
  handles, named in the report.
- It sees the workspace, input trees and handles like any sub-agent. It does not see the parent's
  conversation history.

## Architecture decision (my call)

**sheath owns the delegation provider** instead of using MAF's `BackgroundAgentsProvider`, for
declared and spawned sub-agents alike. Reasons:
- MAF's provider fixes its agent set when it is constructed and keeps its start logic in a
  closure, so adding an agent at runtime would need its private attributes and helpers.
- Nesting and the cascade need control over each task's lifecycle that MAF's provider does not
  expose.
- Task state can live in sheath's `Session` (one per conversation) rather than in each MAF
  `AgentSession`'s state. That removes #9's documented limit that `continue_task` cannot reach a
  task from an earlier AG-UI request.

Constraints on the replacement:
- Keep the six tool names and their observable behaviour (`background_agents_start_task`,
  `..._wait_for_first_completion`, `..._get_task_results`, `..._get_all_tasks`,
  `..._continue_task`, `..._clear_completed_task`) so #9's tests, README and host UIs keep
  working. The wait timeout stays configurable (default 300 s).
- `ConfinedSubagent` stays the run wrapper (cap, bound, spill, tag, activity).
- The parent-wait loop gets a sheath predicate and nudge that read sheath's task state; the
  `_DropLoopNudges` filter stays.
- `release_subagents` and `cancel_subagents` keep their contracts.
- The experimental-feature warning from MAF's provider goes away; the feature stays documented as
  experimental.

## Tools handed down (my call where marked)

- A handed-down plain or `ToolFactory` tool is the creator's own instance, shared.
- A handed-down MCP function is the creator's connected function, sharing its connection (MCP
  sessions accept concurrent calls). It keeps its spill parser. MCP progress notifications from a
  shared server keep the server's original tag (my call: tagging a shared connection per caller
  would need per-call bookkeeping MAF does not offer; documented as a limit).
- Tool-call activity events (`running` / `done` / `failed`) for handed-down tools carry the
  spawned sub-agent's path, as for any sub-agent tool.

## Security argument

A spawned sub-agent's capabilities are a subset of its creator's: bundles ⊆ creator's, host/MCP
tools ⊆ creator's (by name, the same objects; `spawn_subagent` is one of them), limits ≤ creator's
(timeout, round-trips, subtree allowance, depth), so it can spawn only if the creator could and depth
allows, model only from the host allowlist, all on the same `Session`. By induction, nothing in
the tree exceeds the top agent, and the top agent is bounded by the host. The total cap, the
concurrency cap and the per-task bound bound cost. Refusals are tool errors naming the limit, so
an injected request fails visibly and the agent can carry on.

## Left to the builder

- Module layout (a new `sheath/delegation.py` for the provider is the natural home).
- The exact wording of refusal messages and of the tool description, within the rules above.
- How task ids are allocated (per conversation, monotonically increasing, is the obvious choice).
- Test structure, following #9's stub-client patterns.

## Testing (acceptance)

- Per-sub-agent limits: a spawn with `timeout_s` above the creator's bound, `max_round_trips` above
  its budget, `max_subagents` above its remaining allowance, or `max_depth` beyond its remaining
  depth is refused; `max_subagents`/`max_depth` without handing down `spawn_subagent` is refused;
  a sub-agent given `max_subagents=2` spawns two and its third is refused, while a sibling can
  still spawn from the conversation budget; omitted limits inherit the creator's; a
  `timeout_s` shorter than the host's bounds the sub-agent's later `continue_task` too.
- Shared workspace: each spawned sub-agent's folder exists and its instructions name it; agent B's
  `write_file` over a file agent A wrote is refused until B reads it, then succeeds; an agent
  overwriting its own file is unaffected; a single-agent run's file behaviour is unchanged
  (existing file-tool tests pass); two agents saving the same explicit handle id → the second is
  refused; `write_file` is atomic (a reader never sees a partial file).

- A stub parent spawns two sub-agents with different models and tool subsets; both write files in
  the workspace; the parent reads both results. Each sub-agent's client came from the factory
  with its model, and each was offered exactly the tools it was given.
- Refusals, one test each: name taken (declared and spawned), bad name, model not allowed, unknown
  tool, bundle not held, `spawn_subagent` handed down by a creator without it or beyond
  `max_depth`, total cap reached
  (including spawns by nested sub-agents counting against the same cap). Nothing is created by a
  refused call.
- A sub-agent not handed `spawn_subagent` is never offered it. With `max_depth=1`, handing it
  down is refused.
- A sub-agent handed `spawn_subagent` spawns one of its own; status shows `parent/child`; the
  child's tools ⊆ the granted sub-agent's.
- Cascade: a sub-agent that times out while its own sub-agent is running → the child is
  cancelled.
- `continue_task` and `start_task` work on a spawned sub-agent; `continue_task` works across two
  AG-UI requests on the same thread.
- All #9 tests still pass on the sheath-owned provider.
- Reach: a granted sub-agent cannot start a task on a declared role or a sibling, and its
  `get_all_tasks` shows only its own tasks.
- Concurrency under nesting: with `subagent_max_concurrent=2` and `max_depth=2`, two depth-1
  sub-agents that each spawn a child and wait both complete (no deadlock); a waiting agent's
  clock does not advance while it waits.
- Messaging: the parent redirects a running sub-agent and the sub-agent's next model call
  contains the labelled message; sibling to sibling works when permitted and is refused when
  not; a message to an idle agent is delivered at its next task; a message wakes a waiting
  parent; size and count caps refuse; a sub-agent cannot grant a name it cannot message itself.
- Later permission changes: two running siblings start with no permission to message each other;
  the parent calls `set_message_permissions` mid-run, and the next `send_message` between them is
  delivered; after a `deny` it is refused again.
- Working while waiting: a parent starts a slow sub-agent, then makes several calls to its own
  tools; when the sub-agent finishes, the parent's next model call carries the completion notice
  (not the result); the parent reads the result when it chooses, and finishing early is still
  prevented while the task runs.
- One opt-in live test: a real parent spawns two sub-agents on an allowed cheaper model, one
  messages the other, and the parent combines their results.
