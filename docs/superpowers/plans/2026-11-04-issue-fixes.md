# Issues #2–#8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Record, not a work order.** Written after the work was done; every box is checked.

**Goal:** Fix the seven reported issues (jspv/sheath#2 to #8), each on its own branch.

**Specs:**
- `docs/superpowers/specs/2026-11-04-issue-fixes-design.md` (#3 to #8);
- `docs/superpowers/specs/2026-11-04-input-trees-design.md` (#2, written before its implementation).

## Global Constraints

- **TDD.** Write each test first and see it fail.
- **Branches and commits.**
  - Each issue gets a branch from `main` in its own worktree.
  - Nothing is committed or pushed until the owner asks (hard rule 3), and commits are under the
    owner's identity (rule 5).
  - Issue comments and PR text are drafted for the owner to post (rule 6).
- **Commands.**
  - Full suite: `uv run pytest -p no:warnings`.
  - Lint: `uv run ruff check .`.
  - Baseline: 4 failures, from the container environment only.
- **Merging.** Every branch adds an "Unreleased" section to `CHANGELOG.md`. Merging them
  conflicts there, so combine the entries under one heading.

## Branches

| Issue | Branch | Worktree |
|---|---|---|
| #4 | `fix/4-search-scope` | `wt/4-search-scope` |
| #3 | `fix/3-store-option` | `wt/3-store-option` |
| #5 | `fix/5-keep-original-request` | `wt/5-compaction` |
| #6 | `fix/6-mcp-host-payload` | `wt/6-mcp-payload` |
| #8 | `feat/8-tool-loop-limits` | `wt/8-loop-limits` |
| #7 | `feat/7-keep-conversation-workspace` | `wt/7-keep-workspace` |
| #2 | `feat/2-input-trees` | `wt/2-input-trees` |
| docs | `docs/issue-fixes-spec` | `wt/docs` |

**How it landed.** The per-issue branches were developed in separate worktrees, left
uncommitted, and checked both on their own and applied together. At the owner's request they
landed as one branch, `fix/issues-2-8`, with one commit per issue:

| Commit | Issue |
|---|---|
| `41a1e8b` | #4 |
| `1d6fc33` | #3 |
| `0ccfb4d` | #5 |
| `3822e99` | #6 |
| `b7196f1` | #8 |
| `1c390b8` | #7 |
| `117d369` | #2 |

The docs follow in their own commit. The overlaps resolved on the way all kept both sides:
- `session.py`: helpers added at the same spot by #3, #5 and #8, and lines in `create_agent`
  from #8 and #2;
- the README config table: rows from #5 and #8;
- `config.py`: fields from #5 and #8.

The final tree matched the separately tested combination file for file.

## Task 1: #4 `search` scope
- [x] Tests in `tests/tools/test_search.py`:
  - hidden, ignored and plain files on both backends;
  - an include glob never reaches `.sheath/`;
  - a path inside `.sheath/` finds nothing;
  - a nested `.sheath/` is searched.
- [x] `--hidden --no-ignore` and the anchored `!/.sheath/` glob after the caller's glob; an early
  return for paths inside `.sheath/`; the fallback skips `.sheath`.
- [x] README `search` row; CHANGELOG.

## Task 2: #3 `store`
- [x] Tests:
  - an Anthropic-module stub and a host subclass of it get no `store`;
  - an explicit `True` still reaches it;
  - `{"store": None}` removes the key, also through the facade merge.
- [x] `_NO_STORE_CONNECTORS` and `_lacks_store_option`; set `store=False` only when unset and
  supported.
- [x] README storage note; CHANGELOG.

## Task 3: #5 compaction
- [x] `tests/test_compaction.py`:
  - a 12-call run over a 4,000-token input budget keeps the request in the last call;
  - a caller strategy is used;
  - `keep_last_tool_call_groups` reaches the strategy;
  - the facade path;
  - the instructions sentence.
- [x] `SheathConfig.keep_last_tool_call_groups`; `_default_compaction`; core instructions.
- [x] README config table and `create_agent` defaults; CHANGELOG.

## Task 4: #6 MCP host payload
- [x] `tests/test_mcp_host_payload.py`, using the stdio fixture server:
  - the limit becomes 1, and a caller's limit is kept and still warns;
  - a spilled result carries no payload;
  - the omission is logged at debug;
  - tools loaded mid-run are covered too.
- [x] `_drop_mcp_host_payload`, called in `_attach_mcp`; `_MinLimitOmissionToDebug`.
- [x] README MCP section; CHANGELOG.

## Task 5: #8 tool-loop limits
- [x] `tests/test_tool_loop_limits.py`:
  - 45 tool calls finish at a limit of 50 and stop at 40 by default;
  - 5 consecutive errors end the run by default and are survived at 8;
  - a host's own client setting is kept;
  - the facade path;
  - range checks;
  - a client without a tool loop is refused.
- [x] `SheathConfig.tool_loop_max_iterations` / `tool_loop_max_consecutive_errors`;
  `_apply_tool_loop_limits`.
- [x] README "How it works" and config table; CHANGELOG.

## Task 6: #7 keep workspaces
- [x] `tests/test_keep_workspace.py`:
  - keeping on close, and reaping by default;
  - `aopen(keep=)` overrides the config both ways;
  - idle expiry;
  - re-opening resumes the files;
  - `agui_stream`.
- [x] `SheathConfig.keep_conversation_workspaces`; `keep` on `Sheath.aopen` and `SessionManager.aopen`.
- [x] README sessions section and config table; CHANGELOG.

## Task 7: #2 input trees
- [x] Spec written first.
- [x] `tests/test_input_trees.py`, 23 tests covering the issue's acceptance list plus:
  - id conflicts with single-file inputs;
  - persistence across a reopen;
  - dropping a record whose tree is gone;
  - quarantine;
  - a tree staged between turns being announced.
- [x] `sheath/input_trees.py`:
  - the copy, staging and removal;
  - the records file;
  - the instruction sentence and context provider.
- [x] `Session` and `Conversation` methods; the `add_input` guard; `artifacts` excludes `inputs/`;
  the export.
- [x] README "Receiving a directory" and config rows; CHANGELOG.

## Task 8: Roadmap
- [x] Split the v1.1 entry: skills are unblocked upstream but deferred until #2 to #8 merge;
  memory waits on upstream.
