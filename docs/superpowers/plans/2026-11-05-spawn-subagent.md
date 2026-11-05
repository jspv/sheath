# Runtime sub-agents and agent messaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent create its own confined sub-agents at runtime (`spawn_subagent`), nest
them, bound them, keep working while they run, and let agents message each other, without any
sub-agent exceeding its creator.

**Architecture:**
- A new `sheath/delegation.py` replaces MAF's `BackgroundAgentsProvider` with a sheath-owned one.
- Every agent in a conversation is an `AgentNode` in one per-`Session` `Delegation` registry:
  the top agent, declared roles and spawned sub-agents.
- Each node's tools (the six `background_agents_*` tools, `spawn_subagent`, `send_message`,
  `set_message_permissions`) are closures bound to that node, so reach, limits and permissions
  are checked against it.
- `ConfinedSubagent` stays the run wrapper. It gains node-aware limits, giving up its slot while
  waiting, a pausable clock, the cascade and message delivery.

**Tech Stack:** Python 3.12, agent-framework-core 1.22.0 (`ContextProvider`, `AgentLoopMiddleware`,
`MessageInjectionMiddleware` / `agent_framework._sessions.enqueue_messages`, `FunctionMiddleware`),
pytest, `sheath.testing.StubChatClient` (callable steps).

**Spec:** `docs/superpowers/specs/2026-11-05-spawn-subagent-design.md` (and, for everything it
does not change, `docs/superpowers/specs/2026-11-04-subagents-design.md`).

## Global Constraints

- **TDD.** Write each test first and see it fail. Record RED and GREEN.
- **Branch.** Work on `feat/spawn-subagent`. Commit per task with `git add <specific files>`.
  Do not push.
- **No assistant-product names** in tracked files: `git grep -nIiE 'cl[a]ude|\bC[C]\b' -- . ':!uv.lock' ':!.gitignore'`
  prints nothing.
- **Commands:**
  - full suite: `uv run --frozen pytest -p no:warnings -q -o addopts=""` (baseline 1039 passed,
    8 skipped);
  - lint: `uv run --frozen ruff check .`;
  - `uv.lock` must not change;
  - `rm` and `cp` are interactive aliases, so use `command rm -f`;
  - `find sheath tests -name '*.bak' -o -name '*.orig'` prints nothing.
- **Values from the spec, verbatim:**
  - name pattern `^[A-Za-z][A-Za-z0-9_-]{0,63}$`, unique case-insensitively across the whole
    conversation;
  - `SpawnPolicy(models, default_model=None, max_subagents=8, max_depth=3, max_messages=200,
    max_message_chars=4000)`;
  - per-agent folder `agents/<path>/`, with paths joined by `/` (e.g. `lead-2/whois`);
  - message label `[message from <sender path>] `;
  - completion notice
    `[task <id> (<agent path>) <completed|failed> — read it with background_agents_get_task_results]`;
  - status for a delivered message: `StatusEvent(tool="sheath", message="message to <recipient path>", agent=<sender path>)`.
- **Tool names and return strings stay MAF's.** Copy the exact strings from
  `.venv/lib/python3.12/site-packages/agent_framework/_harness/_background_agents.py`, e.g.
  `"Background task {id} started on agent '{name}'."` and
  `"Task {id} finished with status: completed."`. #9's tests and host UIs depend on them.
- **Refusals are tool return strings starting with `Error: `, never exceptions.** Each names the
  limit or rule hit and what is allowed. A refused call creates, counts and changes nothing.
- **Threat model:** the model and fetched content are untrusted. A refusal or status event never
  echoes message text or exception messages from tools.

## Review Focus

1. **A sub-agent reaching past its creator** through any tool: declared roles, siblings, a
   creator's tools not handed down, a model off the allowlist, a limit loosened by omission.
   Every path must be refused. Tests in Tasks 2, 5, 6 and 8.
2. **Deadlock or runaway time under nesting:** a waiting agent that keeps its slot, a clock
   that keeps running while waiting, or a cascade that leaves a grandchild running. Tests in
   Tasks 3 and 6.
3. **Lost updates in the shared workspace:** agent B silently overwriting agent A's file, and a
   single-agent run seeing a change in behaviour. Tests in Task 7.
4. **Message loops and floods:** two agents ping-ponging, oversized messages, messages to agents
   that don't exist or aren't permitted, and permission changes mid-task. Tests in Task 8.
5. **#9 regressions from replacing the provider:** return strings, status events, cancellation
   at turn end and close, AG-UI streaming. Tests in Task 1 (the whole #9 suite) and Task 9.

---

### Task 1: The sheath-owned delegation provider (parity with #9)

Replace MAF's provider for declared sub-agents, with identical observable behaviour.

**Files:**
- Create: `sheath/delegation.py`
- Modify: `sheath/session.py` (`_build_subagents`, `_add_parent_loop`, `release_subagents`,
  Session fields), `sheath/subagents.py` (`ConfinedSubagent` takes a node)
- Test: `tests/test_delegation.py`; existing `tests/test_subagents*.py`

**Interfaces:**
- Produces, in `sheath/delegation.py`:
  - `@dataclass class Limits`:
    - `timeout_s: float | None`
    - `max_round_trips: int | None`
    - `subtree_allowance: int | None`
    - `remaining_depth: int`
  - `class AgentNode`, a plain class with fields:
    - `name: str`, `path: str`, `depth: int`
    - `creator: AgentNode | None`
    - `agent: Any` (a `ConfinedSubagent`; `None` for the top agent)
    - `description: str`, `bundles: tuple[str, ...]`
    - `tool_objects: dict[str, Any]` (handed-down host and MCP tools, by name)
    - `can_spawn: bool`, `limits: Limits`
    - `message_to: set[str]`, `inbox: list[str]`
    - `current_session: Any`
    - `folder: str | None`, `declared: bool`
  - `class TaskRecord`:
    - `id: int`, `owner: AgentNode`, `agent: AgentNode`
    - `description: str`
    - `status: Literal["running", "completed", "failed"]`
    - `result_text: str | None`, `error_text: str | None`
    - `future: asyncio.Task | None`, `sub_session: Any`
  - `class Delegation(session)`, one per `Session`, created lazily as `session.delegation`:
    - `top(name: str) -> AgentNode`: get or create the depth-0 node;
    - `add_node(node: AgentNode) -> None`: raises `ValueError` on a name taken case-insensitively;
    - `node(name: str) -> AgentNode | None`;
    - `start_task(owner, agent_name, input, description) -> str`;
    - `async wait_first(owner, task_ids) -> str`;
    - `results(owner, task_id) -> str`;
    - `all_tasks(owner) -> str`;
    - `continue_task(owner, task_id, text) -> str`;
    - `clear(owner, task_id) -> str`;
    - `running_owned(owner) -> list[TaskRecord]`;
    - `async cancel_all() -> None`.
  - `class DelegationProvider(ContextProvider)`, constructed with `(delegation, node, *,
    wait_timeout_s: int = 300)`. `before_run` sets `node.current_session = session`, then extends
    instructions (sheath's provider note plus a listing of the agents this node can reach) and
    the six tools, as closures over `node`.
  - `tasks_running(node) -> ShouldContinueCallable` and `tasks_running_message(node) ->
    NextMessageCallable`, used by the parent loop.
- `ConfinedSubagent(inner, *, node, session)` replaces the `name` and `description` arguments
  with `node`. `self.name` is `node.path`.

- [ ] **Step 1: Write failing tests** in `tests/test_delegation.py`:
  - `test_declared_subagents_run_on_sheath_provider`: no `BackgroundAgentsProvider` appears in
    the parent agent's context providers; a `DelegationProvider` does.
  - `test_return_strings_match_maf`: the start, wait, results, list, continue and clear tools
    return MAF's strings for the same scripted calls. Copy MAF's strings into the test as
    literals.
  - `test_task_state_survives_new_agent_session`: start task 1 with one `AgentSession`, then call
    `continue_task(1, ...)` with a fresh `AgentSession` on the same conversation; the call is
    accepted (`"Task 1 continued with new input."`).
- [ ] **Step 2: Run them and see them fail.** Then run the existing `tests/test_subagents*.py` to
  record which tests spy on MAF internals.
  `uv run --frozen pytest tests/test_delegation.py tests/test_subagents.py tests/test_subagents_lifecycle.py tests/test_subagents_agui.py tests/test_subagents_api.py -p no:warnings -q -o addopts=""`
- [ ] **Step 3: Implement.**
  - Task state lives in `Delegation`, not in `AgentSession.state`.
  - `start_task` creates `asyncio.create_task(node.agent.run(input, session=sub_session))`.
    The sub-session is created per task and kept on the record for `continue_task`.
  - `cancel_all` and `Session.cancel_subagents` must agree; keep `cancel_subagents`' contract.
    `release_subagents(agent_session)` becomes "cancel this conversation's running tasks";
    its signature stays the same.
  - The parent loop uses `tasks_running(node)` and `tasks_running_message(node)` in place of
    MAF's helpers. Keep `_DropLoopNudges` and `return_final_only=True`.
  - The `wait_timeout_s` default is 300; `background_agents_wait_timeout_seconds` in
    `agent_kwargs` still sets it.
  - Update only those #9 tests that spy on `BackgroundAgentsProvider` (for example
    `release_session` spies) to spy on the sheath equivalent. Update the README sentence about
    `continue_task` across AG-UI requests: it now works.
- [ ] **Step 4: Run the new tests, the #9 tests, then the full suite.** Everything passes; the
  only #9 test edits are the MAF-spy replacements.
- [ ] **Step 5: Commit:** `feat(delegation): sheath-owned delegation provider for declared sub-agents`

---

### Task 2: Reach and completion notices

**Files:**
- Modify: `sheath/delegation.py`
- Test: `tests/test_delegation.py`

**Interfaces:**
- Consumes: `Delegation`, `AgentNode`, `TaskRecord` (Task 1).
- Produces:
  - `Delegation.reachable(owner) -> list[AgentNode]`: for the top agent, the declared roles plus
    the nodes it spawned; for a sub-agent, the nodes it spawned;
  - `Delegation._notify(record) -> None`, called when a task ends;
  - `deliver(node, text) -> None`, a module function: if `node.current_session` belongs to a
    running task or turn, `enqueue_messages(node.current_session, text)`; otherwise append to
    `node.inbox`. `DelegationProvider.before_run` and `ConfinedSubagent.run` drain the inbox into
    the session at start.

- [ ] **Step 1: Write failing tests:**
  - `test_tools_see_only_own_tasks`: a sub-agent's `get_all_tasks`, `get_task_results`, `clear`
    and `continue` on a task it did not start return `"Error: No task found with ID N."`, MAF's
    wording.
  - `test_cannot_start_unreachable_agent`: a node can't start a declared role or a sibling; the
    error lists the agents it can reach.
  - `test_completion_notice_injected_not_result`: a parent starts a slow task, then calls a host
    tool three times. The model call after the task ends contains the notice string exactly, and
    the result text does not appear in any message until `get_task_results`.
  - `test_notice_for_failed_and_timed_out_tasks` says `failed`.
  - `test_inbox_delivered_when_idle_node_starts`: a notice for a node with no running turn is
    seen on its next run.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Key the record lookup by `(owner, id)` semantics. Ids stay unique
  per conversation and increase monotonically.
- [ ] **Step 4: Run the file, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(delegation): per-owner reach and completion notices`

---

### Task 3: Concurrency and time while waiting

**Files:**
- Modify: `sheath/subagents.py` (`ConfinedSubagent`), `sheath/delegation.py` (`wait_first`)
- Test: `tests/test_delegation_waiting.py`

**Interfaces:**
- Consumes: `AgentNode` (Task 1).
- Produces: `AgentNode.waiting() -> AsyncContextManager[None]`. While a `ConfinedSubagent` run
  holds a slot, entering it releases the slot and records the time; leaving it re-acquires the
  slot and moves the run's `asyncio.Timeout` deadline later by the time spent waiting, with
  `bound.reschedule(bound.when() + waited)`. For the top agent, or outside a run, it does nothing.
  `wait_first` runs its wait inside `node.waiting()`.

- [ ] **Step 1: Write failing tests:**
  - `test_waiting_agent_releases_slot`: `subagent_max_concurrent=1`. Declared sub-agent A's host
    tool (a test tool closing over A's node) calls `Delegation.start_task(owner=A_node, "B", ...)`
    and then `async with A_node.waiting():` until B's task is done. B runs and completes; A
    completes. Nesting through `spawn_subagent` arrives in Task 6, so this exercises the primitive
    directly. It would deadlock without the release; guard it with `asyncio.wait_for(..., 5)`.
  - `test_wait_time_not_counted`: `subagent_task_timeout_s=0.5`; A waits 1.0 s on B (B's tool
    sleeps 1.0 s), then finishes. A is not timed out, and B, with its own 0.5 s bound and a 1.0 s
    sleep, is.
  - `test_working_time_still_bounded`: A's own tool sleeps 1.0 s with a 0.5 s bound, so A is
    timed out.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** The slot becomes an explicit acquire/release held by the run, not
  an `async with`, so the wait can hand it back. Keep the existing registration order and the
  cancellation yield.
- [ ] **Step 4: Run the file and `tests/test_subagents*.py`, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(delegation): waiting agents release their slot and pause their clock`

---

### Task 4: `SpawnPolicy` and `spawn_subagent` for the top agent

**Files:**
- Modify: `sheath/subagents.py` (add `SpawnPolicy` and `validate_spawn_policy`),
  `sheath/delegation.py` (the `spawn_subagent` tool), `sheath/session.py`
  (`create_agent(..., spawn=None)`)
- Test: `tests/test_spawn.py`

**Interfaces:**
- Consumes: Tasks 1 to 3. `client_factory` and `validate_subagents`' client identity checks
  (`sheath/subagents.py`).
- Produces:
  - `SpawnPolicy`, exactly as in the spec.
  - `validate_spawn_policy(policy, client_factory) -> None`, which raises `ValueError` for:
    empty `models`; a non-string or empty model; `default_model` not in `models`;
    `max_subagents < 1`; `max_depth < 1`; `max_messages < 1`; `max_message_chars < 1`; no
    factory.
  - `Session.create_agent(..., spawn: SpawnPolicy | None = None)`. With a policy, the top node
    gets `can_spawn=True` and its `Limits` from the host (see the spec's "Per-sub-agent limits").
  - The tool `spawn_subagent(name, instructions, task, model=None, tools=None, bundles=None,
    message_to=None, description="", timeout_s=None, max_round_trips=None, max_subagents=None,
    max_depth=None) -> str`, offered only to nodes with `can_spawn`.
    - It checks every refusal in the spec's table, in the table's order, with the total cap last.
    - On success it creates the client with `client_factory(model or default_model)`. The client
      must be distinct from every other client in the conversation, so reuse the identity checks.
    - It builds the agent with `Session.create_agent(..., subagents=None, spawn=None)`, using the
      handed-down bundles and tools, and wraps it in `ConfinedSubagent`.
    - It creates the folder `agents/<path>/` with `safe_path`, adds the node, increments
      `spawned_total`, and starts the first task through `start_task`.
    - It returns the `start_task` confirmation.
  - The tool's description text, refreshed in `before_run`, lists the allowed models, the default
    model and the remaining budget.

- [ ] **Step 1: Write failing tests**, one per refusal row of the spec table: name taken (declared
  and spawned), bad name, empty instructions or task, model not allowed, unknown tool (the error
  lists the creator's tool names), bundle not held, and the total cap. For each, assert that
  nothing changed: no node, no task, no folder, `spawned_total` unchanged. Plus:
  - `test_spawn_two_with_different_models_and_tools` is the spec's first acceptance test. A
    recording factory sees both models; each sub-agent's client is offered exactly its granted
    tools (record the tool names per model call); both write files that the parent reads.
  - `test_no_policy_no_tool`: without `spawn`, `spawn_subagent` is never offered.
  - `test_policy_validation`: one case per `ValueError` above.
  - `test_tool_description_shows_models_and_budget`.
  - `test_spawned_agent_startable_by_name`: a later `start_task` on the spawned name works.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** A handed-down host tool is the creator's spill-wrapped
  `FunctionTool` instance. A handed-down MCP function is the creator's connected function. Look
  both up in the creator's built tool list by name; never re-wrap or reconnect. Each sub-agent
  gets `_ToolActivity`.
- [ ] **Step 4: Run the file, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(spawn): spawn_subagent and SpawnPolicy for the top agent`

---

### Task 5: Facade plumbing and export

**Files:**
- Modify: `sheath/api.py`, `sheath/manager.py`, `sheath/conversation.py`, `sheath/__init__.py`
- Test: `tests/test_spawn_api.py`

**Interfaces:**
- Consumes: `create_agent(spawn=...)` (Task 4).
- Produces:
  - `Sheath(..., spawn: SpawnPolicy | None = None)`;
  - `aopen`, `asolve`, `Sheath.solve` and module-level `solve` gain `spawn=` with the same
    precedence as `subagents` (a value that is not `None` replaces the default whole);
  - `Conversation.acreate(..., spawn=None)`;
  - `from sheath import SpawnPolicy`.

- [ ] **Step 1: Write failing tests:**
  - the `Sheath` default reaches `aopen`;
  - a per-call value overrides it;
  - `asolve` override;
  - export identity;
  - without an injected client, the built-in factory builds a spawned sub-agent's client for its
    model (monkeypatch `OpenAIChatClient`, as in `tests/test_subagents_api.py`).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement**, following how `subagents` flows (`git show eb37083`).
- [ ] **Step 4: Run the file, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(spawn): spawn= on Sheath, aopen, asolve and solve`

---

### Task 6: Nesting, per-sub-agent limits and the cascade

**Files:**
- Modify: `sheath/delegation.py`, `sheath/subagents.py` (`SubagentSpec.can_spawn`,
  `ConfinedSubagent`), `sheath/status.py` only if `bind_agent` needs the path (it receives
  `node.path`)
- Test: `tests/test_spawn_nesting.py`

**Interfaces:**
- Consumes: Tasks 1 to 5.
- Produces:
  - `spawn_subagent` accepts `"spawn_subagent"` in `tools` (granting `can_spawn`) and the four
    limit arguments;
  - `effective_limits(creator, args) -> Limits | str`, which returns a `Limits` or the
    `"Error: ..."` refusal;
  - `SubagentSpec.can_spawn: bool = False`;
  - a spawn by any node counts against every ancestor's `subtree_allowance` and the
    conversation total;
  - in `ConfinedSubagent.run`'s `finally`, the cascade cancels every running task whose owner is
    this node, and awaits them.

- [ ] **Step 1: Write failing tests:**
  - `test_spawn_tool_handed_down_enables_nesting`: status shows `lead/child`; the child's tools
    are a subset of the lead's; the child's folder is `agents/lead/child/`.
  - `test_not_handed_down_not_offered`.
  - `test_max_depth_refuses_hand_down`: host `max_depth=1`, so handing down is refused.
  - `test_limits_tighten_only`: a `timeout_s`, `max_round_trips`, `max_subagents` or
    `max_depth` above the creator's effective value is refused, one case each.
  - `test_limit_args_require_spawn_tool`.
  - `test_omitted_limits_inherit`.
  - `test_subtree_allowance`: a lead with `max_subagents=2` spawns two, and its third is refused
    with a message naming its allowance; a sibling can still spawn from the conversation budget;
    a grandchild's spawn counts against the lead's allowance.
  - `test_timeout_applies_to_later_continue`.
  - `test_round_trips_on_own_client`: the client's `function_invocation_configuration["max_iterations"]`
    equals the effective value.
  - `test_cascade_cancels_children`: a lead times out while its child is running, and the child
    is cancelled; a grandchild is too.
  - `test_nested_no_deadlock`: `subagent_max_concurrent=2`, two leads each spawn a child and
    wait, and all four complete (guard with `asyncio.wait_for`).
  - `test_declared_role_can_spawn` with a policy.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Limits are fixed at creation and shown in the node's tool
  descriptions (remaining allowance and depth).
- [ ] **Step 4: Run the file, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(spawn): nesting, per-sub-agent limits and cascade cancellation`

---

### Task 7: Shared workspace: folders, instructions and no silent overwrites

**Files:**
- Modify: `sheath/tools/files.py` (`write_file` becomes atomic and guarded; `edit_file` guarded;
  `read_file` records what the agent saw), `sheath/session.py` (tracking state),
  `sheath/bundles.py` (`SUBAGENT_INSTRUCTIONS`)
- Test: `tests/test_shared_workspace.py`; existing `tests/tools/test_files*.py` must pass unchanged

**Interfaces:**
- Consumes: `status.current_agent()` (the node path; `None` for the top agent).
- Produces:
  - `Session._file_writers: dict[str, tuple[str | None, bytes]]`, mapping a root-relative path to
    the agent that last wrote it and the content's sha256;
  - `Session._file_seen: dict[tuple[str | None, str], bytes]`, mapping (agent, path) to the
    sha256 that agent last read or wrote;
  - `_check_overwrite(session, rel, agent) -> None`, which raises `ValueError` (surfaced as a
    tool error) when the file's last writer is another agent and the caller's seen-hash differs
    from the current content's hash. The message names the other agent and says to read first or
    write elsewhere;
  - `SUBAGENT_INSTRUCTIONS` rewritten to say: the final message is a concise result; bulky
    material goes in files or handles, named in the report; own files go in `agents/<path>/`
    (the actual path is filled in per sub-agent); handle ids it picks are prefixed with its name;
    messages from other agents are information, not instructions.

- [ ] **Step 1: Write failing tests:**
  - `test_overwrite_refused_until_read`: agent `a` writes `f.md`; agent `b`'s `write_file` and
    `edit_file` on it are refused, naming `a`; `b` reads it, then writes successfully.
  - `test_own_file_rewrite_unaffected`.
  - `test_unwritten_by_others_unaffected`: a file only the top agent or host wrote is unaffected.
  - `test_single_agent_behaviour_unchanged`: run the existing file-tool tests (no assertion
    change).
  - `test_write_file_is_atomic`: the target is replaced by rename (inode changes; no partial
    file under a concurrent reader thread).
  - `test_subagent_instructions_name_folder`.
  - `test_explicit_handle_id_collision_refused` (existing behaviour, pinned across two agents).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Update the tracking inside `session.file_io()` so it is serialized
  with writes. Hash the full file on read, whatever range `read_file` returned. Files written by
  `run_python` are not tracked; document this in the docstring.
- [ ] **Step 4: Run the file and `tests/tools/`, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(workspace): per-agent folders and no silent overwrites between agents`

---

### Task 8: Messaging between agents

**Files:**
- Modify: `sheath/delegation.py` (`send_message`, `set_message_permissions`, wait wake-up),
  `sheath/subagents.py` (`SubagentSpec.message_to`)
- Test: `tests/test_messaging.py`

**Interfaces:**
- Consumes: `deliver(node, text)` (Task 2), `Delegation.reachable` (Task 2), `AgentNode.message_to`.
- Produces:
  - `send_message(to: str, text: str) -> str`, offered to every node when a policy is set;
  - `set_message_permissions(agent: str, allow: list[str] | None = None, deny: list[str] | None
    = None) -> str`, offered to nodes that have reachable sub-agents;
  - `may_message(sender, recipient_name) -> bool`:
    - the top agent may message any node;
    - a sub-agent may message a name in `message_to`, or its creator unless denied;
  - spawn's `message_to` and later `allow` may only name agents the granter may message itself;
  - `SubagentSpec.message_to: tuple[str, ...] = ()`;
  - `wait_first` returns early with `"A message arrived; check it before waiting again."` when a
    message is delivered to the waiting node; the tasks keep running.

- [ ] **Step 1: Write failing tests:**
  - `test_parent_redirects_running_subagent`: the sub-agent's next model call contains
    `[message from <parent name>] ...`.
  - `test_sibling_message_permitted_and_refused`.
  - `test_permissions_changed_mid_run`: two running siblings are refused at first; after
    `set_message_permissions(allow=...)` a message is delivered; after `deny` it is refused again.
  - `test_message_to_idle_agent_delivered_next_task`.
  - `test_message_wakes_waiting_parent`.
  - `test_size_cap_refused_with_file_hint`.
  - `test_count_cap_refused`.
  - `test_cannot_grant_unmessageable_name`.
  - `test_unknown_recipient_refused`.
  - `test_status_event_without_text`: the event has no message text.
  - `test_messages_framed_as_information`: the label is present.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Check the caps before delivery; a refused send doesn't count.
- [ ] **Step 4: Run the file, then the full suite:** PASS.
- [ ] **Step 5: Commit:** `feat(messaging): send_message and set_message_permissions between agents`

---

### Task 9: AG-UI checks, docs, example and live test

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `examples/subagents_poem.py` (untracked today; rewrite and
  add it)
- Create: `tests/test_spawn_live.py`
- Test: `tests/test_spawn_agui.py`

- [ ] **Step 1: Write failing tests** (`tests/test_spawn_agui.py`):
  - `spawn_subagent` streams as a `TOOL_CALL_*` event;
  - a spawned sub-agent's status streams live with its path;
  - `continue_task` works across two AG-UI requests on one thread (the #9 limit is gone).
- [ ] **Step 2: Run them and see them fail**, then implement any fix they need (expected: none
  beyond Task 1).
- [ ] **Step 3: Docs.**
  - README: a "Runtime sub-agents and messaging (experimental)" section: `SpawnPolicy`, the tool
    and its arguments, limits that only tighten, nesting, reach, working while sub-agents run,
    completion notices, messaging and permissions, shared-workspace rules and the `run_python`
    limit, and the MCP-progress tag limit.
  - Update the confinement bullet and the config table. Remove the "experimental warning from
    MAF" sentence.
  - CHANGELOG entry.
- [ ] **Step 4: Example and live test.**
  - Rewrite `examples/subagents_poem.py` so that a single parent instruction set spawns the
    timer and poet itself under `SpawnPolicy(models=("gpt-5-mini",))`, and the timer messages
    the parent when time is up.
  - `tests/test_spawn_live.py` (skips unless `SHEATH_LIVE=1`): a real parent spawns two
    sub-agents on `gpt-5-mini`, one messages the other, and the parent combines their results.
    Assert spawn and message status events with paths.
  - Run the example once, and the live test once with `SHEATH_LIVE=1`; record the output in the
    report.
- [ ] **Step 5: Full verification, then commit.**
  - Run the full suite, ruff, the name grep and the debris `find`.
  - Commit: `docs(spawn): README, changelog, example and live test`
