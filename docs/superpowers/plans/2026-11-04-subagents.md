# Sub-agents (#9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a host declare named sub-agents that sheath builds over the parent's own `Session`
and hands to MAF's `BackgroundAgentsProvider`, so a sheath agent can delegate without leaving its
confinement.

**Architecture:**
- A new `sheath/subagents.py` holds three things:
  - `SubagentSpec`;
  - spec validation;
  - `ConfinedSubagent`, a wrapper around each built agent that enforces the concurrency cap and
    the wall-clock bound, tags status events and spills results.
- `Session.create_agent` builds each sub-agent by calling its own build path on the same
  `Session`, constructs the provider itself, and keeps it for release.
- Lifecycle hooks in `Session`, `Conversation` and `SessionManager` cancel sub-agent tasks before
  a root is closed or reaped.

**Tech Stack:** Python 3.12, agent-framework-core 1.22.0 (`BackgroundAgentsProvider`,
`AgentResponse`, `Message`), pytest, `sheath.testing.StubChatClient`.

**Spec:** `docs/superpowers/specs/2026-11-04-subagents-design.md`

## Global Constraints

- **TDD.** Write each test first and see it fail.
- **Commits.** Branch `feat/9-subagents` from `main`. Nothing is committed or pushed until the
  owner asks; the commit steps below are where the owner may ask for one.
- **No assistant-product names** in tracked files: run the Task 7 grep before committing.
- **Commands.**
  - Full suite: `uv run pytest -p no:warnings`. Lint: `uv run ruff check .`.
  - Use `--frozen` with `uv run` if the lock drifts; never commit lock drift.
- **Names and defaults:**
  - `SheathConfig.subagent_max_concurrent: int = 2`;
  - `SheathConfig.subagent_task_timeout_s: float | None = 600.0`;
  - `_SUBAGENT_CANCEL_WAIT_S = 30.0`;
  - name pattern `^[A-Za-z][A-Za-z0-9_-]{0,63}$`;
  - handle source `"subagent:<name>"`.
- **Copy:** the provider note, `SUBAGENT_INSTRUCTIONS` and the timeout message are fixed
  verbatim by the spec.
- **Tests run on the local sandbox tier.** `tests/conftest.py` sets
  `SHEATH_SANDBOX_BACKEND=local`.

## Review Focus

1. **A conversation closed while a sub-agent is mid-`run_python` or mid-`write_file`.** The tool
   finishes or is cancelled before the root is removed, and nothing is written into a deleted
   root. Test in Task 4.
2. **The parent model says it's done while a task is still running.** It is re-run until the task
   finishes or times out, and the turn's text is only the final answer. Test in Task 3.
3. **`continue_task` on a finished task.** It goes through the same wrapper, so the cap, bound,
   spill and tag still apply. Test in Task 3.
4. **A host passes its parent client object as a spec's `client`, or reuses one client across
   specs.** This raises, so no sub-agent budget can change the parent's limits. Test in Task 2.
5. **A sub-agent whose result exceeds `max_spill_bytes`.** The task fails with
   `SpillLimitExceeded`'s message, and the parent turn does not crash. Test in Task 3.

---

### Task 1: Status events carry the agent name

**Files:**
- Modify: `sheath/status.py`, `sheath/mcp_status.py`, `sheath/agui.py:19-30`,
  `sheath/cli.py:34-38`
- Test: `tests/test_status.py`, `tests/test_mcp_status.py`, `tests/test_agui.py`,
  `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `StatusEvent.agent: str | None = None`, as the last field;
  - `status.bind_agent(name: str) -> ContextManager[None]`;
  - `status.current_agent() -> str | None`;
  - `install_status_wrappers(bus, tool, server, *, agent: str | None = None)`.

- [ ] **Step 1: Write failing tests**

```python
def test_emit_fills_agent_from_bound_context():
    bus, got = StatusBus(), []
    bus.subscribe(got.append)
    with bind_bus(bus), bind_agent("alpha"):
        report_progress("hi", tool="t")
    report_progress("outside", tool="t")          # unbound: no-op
    assert [(e.tool, e.agent) for e in got] == [("t", "alpha")]

def test_explicit_agent_wins_and_default_is_none():
    bus, got = StatusBus(), []
    bus.subscribe(got.append)
    with bind_agent("alpha"):
        bus.emit(StatusEvent(tool="x", message="m", agent="beta"))
    bus.emit(StatusEvent(tool="x", message="m"))
    assert [e.agent for e in got] == ["beta", None]

def test_bind_agent_restores_previous_value():
    with bind_agent("a"):
        with bind_agent("b"):
            assert current_agent() == "b"
        assert current_agent() == "a"
    assert current_agent() is None
```

Add these alongside the existing tests:
- `tests/test_mcp_status.py`: `test_logging_wrapper_stamps_agent`. Using the file's `_FakeTool`
  and `_LogParams`, `install_status_wrappers(bus, tool, "srv", agent="alpha")` → the emitted
  event has `agent == "alpha"`.
- `tests/test_agui.py`: `status_to_agui(StatusEvent(tool="t", message="m",
  agent="alpha")).value["agent"] == "alpha"`.
- `tests/test_cli.py`: the verbose sink writes `→ [alpha] t: m` for an event with an agent, and
  `→ t: m` without one.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_status.py tests/test_mcp_status.py tests/test_agui.py tests/test_cli.py -p no:warnings`
Expected: `ImportError: cannot import name 'bind_agent'`, then failures on the missing `agent`
field.

- [ ] **Step 3: Implement**

`status.py`:
- `_current_agent: ContextVar[str | None]`;
- `bind_agent`, restoring with `set()` as `bind_bus` does; its docstring says why;
- `StatusBus.emit` fills `agent` from `current_agent()` in the same `dataclasses.replace` call,
  only when `event.agent is None`.

Elsewhere:
- `mcp_status`: pass `agent=` through to every `StatusEvent` it builds;
- AG-UI: add `"agent": event.agent` to the value;
- CLI: add the `[agent] ` prefix only when the field is set.

- [ ] **Step 4: Run the four files and see them pass, then run the full suite**

Run: `uv run pytest -p no:warnings`
Expected: all pass. The baseline is 831 passed, 10 skipped, plus the new tests.

- [ ] **Step 5: Commit**

```bash
git add sheath/status.py sheath/mcp_status.py sheath/agui.py sheath/cli.py tests/test_status.py tests/test_mcp_status.py tests/test_agui.py tests/test_cli.py
git commit -m "feat(status): tag status events with the emitting agent"
```

---

### Task 2: `SubagentSpec`, config fields and validation

**Files:**
- Create: `sheath/subagents.py`
- Modify: `sheath/config.py` (two fields after `tool_loop_max_consecutive_errors`, each with a
  comment), `sheath/bundles.py` (add `SUBAGENT_INSTRUCTIONS`)
- Test: `tests/test_subagents_spec.py`

**Interfaces:**
- Produces:
  - `SubagentSpec`, exactly as in the spec's "Public surface";
  - `validate_subagents(specs: Mapping[str, SubagentSpec], *, parent_bundles: tuple[str, ...],
    parent_client: Any, client_factory: Callable[[str], Any] | None, config: SheathConfig) ->
    dict[str, Any]`. It returns `{name: resolved_client}` in insertion order. Each client is
    `spec.client`, or else `client_factory(spec.model or config.model)`. The factory is called
    once per spec that has no client, and only after every structural check has passed;
  - `bundles.SUBAGENT_INSTRUCTIONS: str`.

- [ ] **Step 1: Write failing tests**, one per row of the spec's validation table:

```python
def test_spec_has_no_subagents_field():
    with pytest.raises(TypeError):
        SubagentSpec(description="d", subagents={})

def test_non_spec_value_rejected():
    with pytest.raises(TypeError, match="SubagentSpec"):
        _validate({"a": {"description": "d"}})

@pytest.mark.parametrize("name", ["", "1abc", "has space", "a" * 65, "ünï"])
def test_bad_names(name): ...                       # ValueError, message names the pattern

def test_case_insensitive_duplicate(): ...          # {"Alpha": s, "alpha": s} -> ValueError
def test_bundle_outside_parent(): ...               # parent ("files",), spec ("web",) -> ValueError naming "web"
def test_parent_empty_bundles_means_all(): ...      # parent (), spec ("web",) -> ok
def test_budget_below_one(): ...                    # max_round_trips=0 -> ValueError
def test_model_and_client_both_set(): ...           # ValueError
def test_parent_client_reused_rejected(): ...       # spec.client is parent_client -> ValueError   (Review Focus 4)
def test_client_shared_across_specs_rejected(): ... # same object in two specs -> ValueError     (Review Focus 4)
def test_no_client_no_factory(): ...                # ValueError naming "client=" and "client_factory"
def test_factory_gets_spec_model_else_config_model(): ...   # recorded calls == ["m-small", SheathConfig().model]
def test_spec_client_wins_over_factory(): ...       # the factory is not called for that spec
def test_config_bounds(): ...                       # subagent_max_concurrent=0 / subagent_task_timeout_s=0 -> ValueError
def test_factory_not_called_when_validation_fails(): ...
```

Here `_validate(specs, **kw)` is a local helper that calls `validate_subagents` with defaults:
`parent_bundles=("files",)`, a `StubChatClient` as the parent client, a recording factory that
returns a new `StubChatClient` for each model, and `SheathConfig()`.

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_subagents_spec.py -p no:warnings`
Expected: `ModuleNotFoundError: sheath.subagents`.

- [ ] **Step 3: Implement `SubagentSpec` and `validate_subagents` in `sheath/subagents.py`**

- Add the two config fields.
- Use `bundles.selected_bundles` for the subset check.
- `SUBAGENT_INSTRUCTIONS` reads: "You are a sub-agent: another agent delegated this task to
  you, and your final message is your report back to it. Files and handles you create are
  shared with it, so name the ones that matter in your report rather than pasting their
  contents."

- [ ] **Step 4: Run and see them pass**

Run: `uv run pytest tests/test_subagents_spec.py tests/test_config.py tests/test_bundles.py -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sheath/subagents.py sheath/config.py sheath/bundles.py tests/test_subagents_spec.py
git commit -m "feat(subagents): SubagentSpec and its validation"
```

---

### Task 3: Build confined sub-agents in `create_agent`

**Files:**
- Modify:
  - `sheath/subagents.py` (add `ConfinedSubagent`, `_ToolActivity` and `provider_instructions`);
  - `sheath/session.py` (`create_agent`, `_attach_mcp`, new Session fields);
  - `sheath/spill.py:_maybe_spill`;
  - `sheath/testing.py`.
- Test: `tests/test_subagents.py`

**Interfaces:**
- Consumes:
  - `validate_subagents` and `SUBAGENT_INSTRUCTIONS` (Task 2);
  - `bind_agent` and `StatusEvent.agent` (Task 1).
- Produces:
  - `Session.create_agent(..., subagents: Mapping[str, SubagentSpec] | None = None,
    client_factory: Callable[[str], Any] | None = None)`;
  - `Session.subagent_names: tuple[str, ...]` (default `()`);
  - `Session.subagents_running -> int`;
  - `Session._subagent_tasks: set[asyncio.Task]`;
  - `Session._subagent_providers: list`;
  - `_maybe_spill(session, tool_name, result, *, source: str | None = None)`, where the default
    source stays `f"tool:{tool_name}"`;
  - `_attach_mcp(tool, *, agent: str | None = None)`;
  - `StubChatClient` script steps may be `Callable[[list[Message]], Content]`, called with that
    turn's messages.
  - `_ToolActivity`, a MAF `FunctionMiddleware` that emits `running`, `done` or
    `failed: <ExceptionType>` with `tool=context.function.name`.
  - `_add_parent_loop(maf_kwargs) -> None`, which inserts sheath's loop middleware. Task 6 adds
    `_DropLoopNudges` ahead of the loop inside this helper.

- [ ] **Step 1: Write failing tests.** Helpers at the top of the file:
  - `parent(script)`: a `StubChatClient` with callable steps;
  - `start(name, i)`: `tool_call("background_agents_start_task", {"agent_name": name, "input":
    "go", "description": name}, f"s{i}")`;
  - `wait(ids, i)`, `results(tid, i)`, `cont(tid, text, i)`;
  - `echo_results`: a final step returning `text(" | ".join(...))` over every `function_result`
    in the messages;
  - `capture(into)`: a step that appends each `function_result`'s `(result, exception)` to
    `into`, then returns `text("done")`.

```python
def test_two_subagents_write_files_and_report(tmp_path):     # acceptance 1
    # parent: start a, start b, wait [1,2], wait [2], results 1, results 2, echo_results
    # sub a: write_file("a.txt","A") then text("report-a"); sub b likewise
    # -> (root/"a.txt").read_text()=="A", same for b; final text contains "report-a" and "report-b"
    # -> sess.subagent_names == ("a", "b")

def test_long_result_arrives_as_handle(tmp_path):             # acceptance 2
    # spill_threshold_bytes=64, sub returns text("x"*500)
    # -> the results tool's text parses as JSON with {"id","kind","path"};
    #    sess.handles[that id]["source"] == "subagent:a"

def test_result_over_spill_cap_fails_task(tmp_path):          # Review Focus 5
    # max_spill_bytes=100, sub returns 500 chars -> results text contains "spill cap";
    # the parent run returns normally

def test_write_outside_root_fails(tmp_path):                  # acceptance 3
    # sub: write_file("../out.txt","x") then capture(seen)
    # -> not (tmp_path/"out.txt").exists(); "path escapes root" in seen[0][1]

def test_background_agents_still_refused(tmp_path):           # acceptance 5
    with pytest.raises(TypeError, match="background_agents"):
        await sess.create_agent(StubChatClient([text("x")]), background_agents=[object()])

def test_subagent_tool_status_carries_name(tmp_path):         # acceptance 6
    # host tool `ping()` calls report_progress("p", tool="ping"); subscribe to sess
    # -> events with tool=="ping" all have agent=="a"; a "started" and a "finished" event
    #    (tool=="sheath", agent=="a") exist; the parent's own tool events have agent None

def test_every_subagent_tool_call_reports_activity(tmp_path):
    # sub calls write_file (ok) then write_file("../x") (raises)
    # -> (tool, message, agent) sequence includes ("write_file","running","a"),
    #    ("write_file","done","a"), ("write_file","running","a"),
    #    ("write_file","failed: PathEscapesRootError","a")
    # -> no failed message contains "../x" (the exception text is never echoed)
    # -> the parent's background_agents_* calls emit no activity events

def test_cap_one_serializes(tmp_path):                         # acceptance 7
    # subagent_max_concurrent=1; host async tool `step(tag)` appends ("start",tag),
    # sleeps 0.05, appends ("end",tag)
    # -> log is [start x, end x, start y, end y] for some order x, y

def test_task_timeout_fails_with_message(tmp_path):
    # subagent_task_timeout_s=0.05; sub's tool sleeps 2s
    # -> results text contains "ran past subagent_task_timeout_s"

def test_continue_task_goes_through_wrapper(tmp_path):        # Review Focus 3
    # spill_threshold_bytes=64; start, wait, cont(1,"more"), wait, results; sub's 2nd reply is 500 chars
    # -> second result is a handle summary; status "started" is emitted twice for agent "a"

def test_tool_factory_built_once_per_subagent(tmp_path):
    # two create_agent calls on two sessions, spec tools=[tool_factory(build)] -> build called twice

def test_subagent_mcp_connected_tagged_and_closed(tmp_path):
    # FakeMCPTool from tests/test_mcp_wiring.py in spec.tools
    # -> connected after create_agent, closed after the session exits

def test_budget_applied_to_own_client_only(tmp_path):
    # spec(client=sub_stub, max_round_trips=2)
    # -> sub_stub.function_invocation_configuration["max_iterations"] == 2;
    #    the parent client's max_iterations is unchanged

def test_parent_rerun_until_tasks_finish(tmp_path):          # Review Focus 2
    # the sub's tool sleeps 0.2s; parent script: start, text("finished early"), wait [1],
    # results 1, echo_results
    # -> the agent.run(..., session=agent.create_session()) text contains "report-a" and does not
    #    contain "finished early"; a parent message holds MAF's "still have 1 background task" nudge

def test_caller_loop_replaces_sheaths(tmp_path):
    # agent_kwargs loop_should_continue=lambda **kw: False -> no sheath AgentLoopMiddleware in
    # agent.middleware; and without it, exactly one, placed before the caller's middleware

def test_caller_loop_max_iterations_used(tmp_path):
    # loop_max_iterations=3 in maf_kwargs -> sheath's AgentLoopMiddleware has max_iterations 3

def test_subagent_cannot_delegate(tmp_path):
    # the sub-agent's script calls background_agents_start_task -> MAF reports the tool is unknown;
    # capture shows an error and no task runs

def test_stream_true_refused(): ...   # ConfinedSubagent.run(stream=True) -> ValueError
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_subagents.py -p no:warnings`
Expected: `TypeError: unexpected keyword argument 'subagents'`, or a failed import of
`ConfinedSubagent`.

- [ ] **Step 3: Implement**

- **`StubChatClient`:** in `_inner_get_response`, if the step at this turn is callable, call it
  with `messages` and use the `Content` it returns.
- **`ConfinedSubagent(inner, *, name, description, session)`:** `run` follows spec steps 1 to 7
  exactly.
  - The semaphore is created lazily on the session as `session._subagent_slots`, because it must
    be created inside the running loop.
  - Registration uses `asyncio.current_task()`: add it before acquiring the slot, discard it in
    `finally`.
  - A spilled dict becomes `json.dumps(summary)`.
- **`create_agent`:**
  1. After the refused-kwargs check, pop `background_agents_wait_timeout_seconds` from
     `maf_kwargs`.
  2. If `subagents`, call `validate_subagents`, then build each sub-agent with
     `self.create_agent(sub_client, agent_instructions=spec.instructions,
     tools=expand_tools(spec.tools), bundles=spec.bundles or bundles, name=name,
     _mcp_agent=name, _harness_extra=SUBAGENT_INSTRUCTIONS)`. These are private keyword-only
     parameters, not part of `**maf_kwargs`.
  3. Apply each budget with `_apply_tool_loop_limits(sub_client,
     dataclasses.replace(self.config, tool_loop_max_iterations=spec.max_round_trips,
     tool_loop_max_consecutive_errors=None))`.
  4. Wrap each agent, construct `BackgroundAgentsProvider(wrapped,
     instructions=provider_instructions(), wait_timeout_seconds=...)` (MAF's default when not
     given), append it to `context_providers` and to `self._subagent_providers`, and set
     `self.subagent_names`.
  5. Call `_add_parent_loop(maf_kwargs)`. Unless `loop_should_continue` is in `maf_kwargs`, it
     pops `loop_max_iterations` and prepends the spec's `AgentLoopMiddleware(...)` to
     `maf_kwargs["middleware"]`, keeping the caller's entries after it.
  6. Each sub-agent's build receives `middleware=[_ToolActivity()]` through its `maf_kwargs`.
- **Imports:** `expand_tools` is imported locally from `.api`.

- [ ] **Step 4: Run the new file and the existing `create_agent` and MCP tests**

Run: `uv run pytest tests/test_subagents.py tests/test_session_create_agent.py tests/test_mcp_wiring.py tests/test_spill.py -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sheath/subagents.py sheath/session.py sheath/spill.py sheath/testing.py tests/test_subagents.py
git commit -m "feat(subagents): build confined sub-agents for MAF's background-agents provider"
```

---

### Task 4: Lifecycle: nothing outlives its workspace

**Files:**
- Modify: `sheath/session.py` (`aclose`, `cancel_subagents`, `release_subagents`, root-removal
  helper), `sheath/conversation.py` (`aask`, `astream_agui`, `aclose`)
- Test: `tests/test_subagents_lifecycle.py`

**Interfaces:**
- Consumes: `Session._subagent_tasks`, `Session._subagent_providers` and
  `Session.subagents_running` (Task 3).
- Produces:
  - `async Session.cancel_subagents() -> None`, bounded by `_SUBAGENT_CANCEL_WAIT_S`; it logs
    any task still pending;
  - `async Session.release_subagents(agent_session) -> None`;
  - `_remove_root_locked(root: Path, lock) -> None`, a sync function run with
    `asyncio.to_thread`.

- [ ] **Step 1: Write failing tests**

```python
def test_session_close_cancels_running_subagent(tmp_path):
    # sub's async tool: try: await asyncio.sleep(5); write marker  finally: flag["cancelled"]=True
    # parent: start, then text("bye") without waiting; after run, exit the session
    # -> flag set; no marker; the exit takes well under 5s

def test_conversation_close_releases_and_reaps_after_cancel(tmp_path):    # Review Focus 1
    # Conversation.acreate with a spec; aask (the parent starts a task and returns)
    # spy on BackgroundAgentsProvider.release_session (monkeypatch) -> called with conv.agent_session
    # -> root gone; marker never written, also after 0.2s more

def test_reap_waits_for_in_flight_sync_file_tool(tmp_path):              # Review Focus 1
    # sub's sync tool holds session.io_lock, sleeps 0.3s, writes a file; close mid-tool
    # -> close returns only after the tool finished; root does not exist afterwards

def test_turn_end_cancels_leftover_tasks(tmp_path):
    # agent_kwargs loop_max_iterations=1, so the re-run cap is hit; the sub sleeps 5s in a
    # try/finally that sets a flag
    # -> aask returns quickly; the flag is set; session.subagents_running == 0; no marker file

def test_agui_disconnect_cancels_tasks(tmp_path):
    # astream_agui closed after its first event (aclosing) while a task runs -> flag set

def test_cleanup_root_uses_locked_removal(tmp_path): ...   # SheathConfig(cleanup=True): root gone after exit
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_subagents_lifecycle.py -p no:warnings`
Expected: the marker gets written or the flag is unset; `release_session` is not called.

- [ ] **Step 3: Implement, in the spec's order**

- **`Session.aclose`:**
  1. `await self.cancel_subagents()`;
  2. close MCP;
  3. if `cleanup`, `await asyncio.to_thread(_remove_root_locked, self.root, self.io_lock)`.
- **`Conversation.aask` and `astream_agui`:** `await self.session.cancel_subagents()` in a
  `finally`, inside the turn lock.
- **`Conversation.aclose`:** `await self.session.release_subagents(self.agent_session)` before
  `session.__aexit__`, then reap via `_remove_root_locked` in a thread.

- [ ] **Step 4: Run lifecycle, conversation, manager and keep-workspace tests**

Run: `uv run pytest tests/test_subagents_lifecycle.py tests/test_conversation.py tests/test_manager.py tests/test_keep_workspace.py tests/test_agui.py -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sheath/session.py sheath/conversation.py tests/test_subagents_lifecycle.py
git commit -m "feat(subagents): cancel sub-agent tasks before a workspace is closed or reaped"
```

---

### Task 5: Facade plumbing and export

**Files:**
- Modify: `sheath/api.py`, `sheath/conversation.py:acreate`, `sheath/manager.py`
  (`aopen`, `_create`), `sheath/__init__.py`
- Test: `tests/test_subagents_api.py`

**Interfaces:**
- Consumes: `Session.create_agent(subagents=, client_factory=)` and `Session.subagent_names`
  (Task 3).
- Produces:
  - `Sheath(..., subagents: Mapping[str, SubagentSpec] | None = None,
    client_factory: Callable[[str], Any] | None = None)`;
  - `aopen`, `asolve` and `solve` (method and module-level) gain
    `subagents: Mapping[str, SubagentSpec] | None = None`;
  - `Conversation.acreate(..., subagents=None, client_factory=None)`;
  - `Sheath._subagent_client_factory() -> Callable[[str], Any] | None`. It returns the
    caller's `client_factory` if given; otherwise, when no `client` was injected,
    `lambda model: OpenAIChatClient(model=model, env_file_path=".env")`; otherwise `None`;
  - `from sheath import SubagentSpec`.

- [ ] **Step 1: Write failing tests**

```python
def test_sheath_default_used_by_aopen(tmp_path):          # conv.session.subagent_names == ("a",)
def test_aopen_override_replaces_default(tmp_path):        # Sheath(subagents=A).aopen(subagents=B) -> ("b",)
def test_empty_dict_override_means_none(tmp_path):         # aopen(subagents={}) -> ()
def test_asolve_override(tmp_path):                        # captured through a spy on Session.create_agent kwargs
def test_injected_client_without_factory_or_spec_client_raises(tmp_path):   # ValueError from aopen
def test_client_factory_called_per_subagent_with_model(tmp_path):
    # Sheath(client=parent_stub, client_factory=rec) with specs {"a": model "m1", "b": no model}
    # -> rec saw ["m1", config.model]; each sub-agent's client is distinct from parent_stub
def test_default_factory_when_no_client_injected(tmp_path, monkeypatch):
    # monkeypatch agent_framework.openai.OpenAIChatClient with a recorder returning stubs
    # -> called with model="m1" for the sub-agent
def test_subagentspec_exported():
    import sheath; assert sheath.SubagentSpec is sheath.subagents.SubagentSpec
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_subagents_api.py -p no:warnings`
Expected: `TypeError: unexpected keyword argument 'subagents'`.

- [ ] **Step 3: Implement**

Thread `subagents`, resolved as `override if override is not None else self._subagents`, and
`client_factory=self._subagent_client_factory()` through `asolve`, `SessionManager._create` and
`Conversation.acreate` into `create_agent`. `agui_stream` takes the `Sheath` default through
`aopen`.

- [ ] **Step 4: Run the API, agent-instructions, agent-kwargs and manager tests**

Run: `uv run pytest tests/test_subagents_api.py tests/test_api.py tests/test_agent_instructions.py tests/test_agent_kwargs.py tests/test_manager.py -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sheath/api.py sheath/conversation.py sheath/manager.py sheath/__init__.py tests/test_subagents_api.py
git commit -m "feat(subagents): subagents= on Sheath, aopen, asolve and solve"
```

---

### Task 6: AG-UI: live status and a clean stream

**Files:**
- Modify: `sheath/agui.py:merge_status`, `sheath/subagents.py` (add `_DropLoopNudges`),
  `sheath/session.py:_add_parent_loop`
- Test: `tests/test_agui.py`, `tests/test_subagents_agui.py`. Both need the `agui` extra; skip
  with `pytest.importorskip("agent_framework_ag_ui")`, as the existing AG-UI tests do.

**Interfaces:**
- Consumes:
  - `_add_parent_loop` (Task 3);
  - `Conversation.astream_agui` with turn-end cancel (Task 4);
  - `subagents=` on `Sheath` (Task 5).
- Produces:
  - `_DropLoopNudges(AgentMiddleware)`;
  - `merge_status`, with an unchanged signature.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agui.py
async def test_status_yielded_while_source_is_blocked():
    # source: yields "a", then awaits an Event before yielding "b"; meanwhile a task emits on
    # the bus and then sets the Event
    # -> the yielded order is ["a", <status>, "b"]; the status arrives before "b" is produced
async def test_early_close_cancels_pending_source_read(): ...   # aclose mid-wait: no pending task left, subscriber removed

# tests/test_subagents_agui.py, through Sheath(...).agui_stream with a stub parent and one sub-agent
def test_subagent_status_streams_before_wait_result(tmp_path):
    # the sub's tool emits report_progress then sleeps 0.3s
    # -> the CUSTOM sheath.status event (agent "a") precedes the TOOL_CALL_RESULT of
    #    background_agents_wait_for_first_completion
def test_nudge_absent_from_text_and_snapshot(tmp_path):
    # parent script: start, text("finished early"), wait, results, text("final")
    # -> no TEXT_MESSAGE_CONTENT delta and no MESSAGES_SNAPSHOT message contains
    #    "background task(s) running"; "final" is present
def test_nonstreaming_run_unaffected(tmp_path):
    # agent.run(...) (non-streaming) with _DropLoopNudges present -> text == "final"
def test_continue_task_across_agui_requests(tmp_path):
    # request 1 starts task 1 and waits for it; request 2 on the same threadId calls continue_task(1, "more")
    # -> assert the observed outcome, either "continued" or "No task found", and name it in the
    #    test's docstring; the README sentence in Task 7 follows it
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_agui.py tests/test_subagents_agui.py -p no:warnings`
Expected: the status arrives after "b"; the nudge text is found. The `continue_task` test is
written once the outcome has been observed, not guessed.

- [ ] **Step 3: Implement**

- **`merge_status`:**
  - keep one `asyncio.ensure_future(source.__anext__())`;
  - each step, `asyncio.wait({pending, queue_get}, return_when=FIRST_COMPLETED)`;
  - yield status as it lands, and source events in order;
  - on `StopAsyncIteration`, drain the queue;
  - in `finally`, cancel whichever task is still pending, then unsubscribe.
- **`_DropLoopNudges`:** as in the spec's "AG-UI" section.
- **`_add_parent_loop`:** inserts `_DropLoopNudges()` immediately before the loop middleware.

- [ ] **Step 4: Run the AG-UI files, then the full suite**

Run: `uv run pytest -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sheath/agui.py sheath/subagents.py sheath/session.py tests/test_agui.py tests/test_subagents_agui.py
git commit -m "feat(agui): stream sub-agent status live and keep loop nudges out of the chat"
```

---

### Task 7: Docs, live test and full verification

**Files:**
- Modify: `README.md`, `CHANGELOG.md` (Unreleased)
- Create: `tests/test_subagents_live.py`

- [ ] **Step 1: Write the live test.** Skip it unless `SHEATH_LIVE=1`, matching
  `tests/test_live.py`. A real `Sheath` with
  `subagents={"counter": SubagentSpec(description="counts words in a workspace file",
  bundles=("files", "code"))}` is asked to write a short text file and have `counter` report its
  word count. Assert the final text contains the correct count and that
  `"background_agents_start_task"` was called, seen through an `on_status` or tool-call spy.

- [ ] **Step 2: README**

- Add a Guides section, "Sub-agents (experimental)": the example `subagents=` with sub-agents
  on two different models, the fields, `client_factory`, the two config rows, and the precedence
  rules. Also cover:
  - parents wait for their tasks, and leftover tasks are cancelled at turn end;
  - hosts with long sub-agent tasks should set generous parent limits
    (`tool_loop_max_iterations`, `loop_max_iterations`);
  - the status `agent` field and the per-tool `running`/`done`/`failed` events;
  - an "AG-UI" paragraph: delegation streams as tool calls and sub-agent activity as
    `sheath.status` events with `agent`; status is live during waits; the
    `continue_task`-across-requests behaviour Task 6 observed.
- In "Security and confinement", add the guarantee: a sub-agent can do nothing the parent could
  not. State the residual risk: reports are model text, treated as data.
- Add the AG-UI `agent` key to "Live status updates".
- Add two config table rows.
- Remove the matching roadmap item, if one exists.

- [ ] **Step 3: CHANGELOG**

Under "Unreleased", add a "### Sub-agents (experimental)" entry. Mark the `StatusEvent.agent`
field as additive.

- [ ] **Step 4: Full verification**

Run: `uv run pytest -p no:warnings && uv run ruff check . && git grep -nIiE 'cl[a]ude|\bC[C]\b' -- . ':!uv.lock'; find sheath tests -name '*.bak' -o -name '*.orig'`
Expected: the suite passes; ruff is clean; the grep and find print nothing.

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md tests/test_subagents_live.py
git commit -m "docs(subagents): README guide, confinement note, changelog"
```
