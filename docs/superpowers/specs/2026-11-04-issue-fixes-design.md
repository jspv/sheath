# Issue fixes #3–#8 — design

Date: 2026-11-04
Status: implemented. This was written after implementation, from the decisions taken during
it. Each fix was built test-first on its own branch. The larger issue, jspv/sheath#2, has its own
spec (`2026-11-04-input-trees-design.md`).

Issues: jspv/sheath#3, #4, #5, #6, #7, #8.

## #4 — `search` sees exactly what `list_files` lists

**Problem.** ripgrep ran with its defaults. It skipped hidden paths and honored any
`.gitignore`, `.ignore` or `.rgignore` in the workspace, so a checked-out project hid its ignored
and dotfile paths from `search` while `list_files` showed them. The Python fallback searched
`.sheath/`, which `list_files` skips.

**Decision.**
- ripgrep runs with `--hidden --no-ignore`, plus `-g '!/.sheath/'` placed *after* the caller's
  glob, so it wins over an include glob.
- The leading `/` anchors the exclusion to the directory ripgrep runs in, the session root. A
  project's own nested `.sheath/` is still searched.
- ripgrep searches a path given explicitly even when a glob excludes it, so a search whose
  `path` points into `.sheath/` returns no results before either backend runs.
- The fallback skips files whose first root-relative part is `.sheath`.

**Verified.**
- ripgrep applies `.gitignore` only inside a git repository, but always applies `.ignore`. The
  test writes both, so it fails before the fix wherever the suite runs.
- `.scripts/` (inline scripts) is hidden, so it is now searched. That matches `list_files`.

## #3 — `store` only for connectors that have it

**Problem.** `create_agent` sent `store=False` to every client. The Anthropic connector forwards
every non-`None` option to its SDK, which rejects `store`.

**Decision.**
- `store=False` is added only when the caller set no `store` key and the client's class, or any
  base class, does not come from a package in `_NO_STORE_CONNECTORS = ("agent_framework_anthropic",)`.
- An explicit caller value always wins.

**Finding.**
- MAF already drops options whose value is `None` before calling the client, and the facade's
  `default_options` merge already keeps `None`. So `{"store": None}` worked as an opt-out
  before the fix; it was just untested. Tests now pin it at `create_agent` and through the
  facade merge.

## #5 — Compaction keeps the original request

**Problem.** `create_harness_agent` builds `ContextWindowCompactionStrategy` with
`preserve_first_user_group=False`. Past 80% of the input budget, truncation dropped the oldest
group, which in a one-shot task is the task itself.

**Decision.**
- `create_agent` builds the strategy itself, with `preserve_first_user_group=True` and
  `keep_last_tool_call_groups` from the new `SheathConfig` field (default 4, MAF's default).
- It uses one instance for both phases, as MAF does.
- A phase the caller supplied is left alone, and `disable_compaction=True` is honored.
- The core instructions now say that older tool results may be collapsed, and that findings
  belong in a handle or a workspace file.

**Finding.** `create_harness_agent` builds its own default strategy whenever the token budgets
are set, even when the caller supplies strategies, and then uses the caller's. A test that
counts constructions has to look for sheath's instance among them.

## #6 — Hidden MCP results do not count toward compaction

**Problem.** MAF keeps a raw copy of each MCP result, up to `max_host_payload_size_bytes`
(1 MiB by default), in the result content's `additional_properties`. Compaction's estimate
counts it, so spilled results still weighed as much as the raw page.

**Decision.**
- In `_attach_mcp`, a tool whose limit is still MAF's default is set to `1`, the smallest value
  MAF accepts.
- A logging filter on `agent_framework._mcp` downgrades MAF's per-call "Omitting MCP ..."
  warning to debug, but only when it reports that 1-byte limit. A limit a caller chose still
  warns.
- The filter is installed once, however many tools are attached.

**Risk checked.** The same limit governs result `_meta`. sheath's progress tracking uses request
`_meta`, and the MCP status tests pass.

## #8 — Tool-loop limits are sheath settings

**Problem.** MAF ends a turn after 40 round-trips and after 3 consecutive tool errors, and sheath
exposed neither limit.

**Decision.**
- MAF reads both only from the client's `function_invocation_configuration`, so `create_agent`
  writes them there, from `SheathConfig.tool_loop_max_iterations` and
  `tool_loop_max_consecutive_errors`.
- **Deviation from the issue:** both fields default to `None`, meaning the client's own setting,
  not to 40 and 3. With hard defaults, sheath would reset a limit a host had already set on its
  client, which is exactly the workaround the issue describes. A fresh client still stops at 40
  and 3.
- Out-of-range values raise `ValueError`, and so does a limit on a client without
  `function_invocation_configuration`.

## #7 — Keep a conversation's workspace

**Problem.** `SessionManager._create` hard-coded `reap_on_close=True`.

**Decision.**
- `SheathConfig.keep_conversation_workspaces` (default `False`), plus `aopen(..., keep=None)`
  to override it per conversation.
- The setting flows to `Conversation.reap_on_close`, which already existed for the one-shot
  `keep=True` path. `agui_stream` follows the config.

## Testing

Each fix has a dedicated test module or section, written first and seen to fail before the fix.
Each branch's full suite passes except the four tests that fail only in the cloud container:
- the tests run as root, so `chmod 0` does not block reads;
- the two grandchild-kill tests;
- podman is not installed.
