# Sub-agent write confinement and sub-agent middleware — Implementation Plan

**Goal:** (1) a sub-agent writes only to its own folder and to paths its creator granted;
(2) the host can give every sub-agent its own middleware, so a policy or audit middleware on the
parent cannot be escaped by spawning a sub-agent.

## 1. Write confinement

**Rule.** The top agent writes anywhere it could before. A sub-agent (declared or spawned) writes
only under its folder `agents/<path>/` (which holds its own sub-agents' folders, so a creator
can write into them) and under the paths in its `write_to`. Reads are unchanged: the whole
workspace.

**Where it is enforced.** `write_file`, `edit_file` and `publish_handle` (which writes
`outputs/<name>`), on the path `safe_path` resolved to, so `..` and symlinks cannot leave the
scope. The current agent comes from `CURRENT_RUN` (set for every tool call of a sub-agent run);
no run means the top agent. A refusal raises `ValueError` naming the paths it may write.
Case aliases on case-insensitive filesystems can only cause a refusal, never a wider grant.

**Grants.** `write_to: list[str]` on `spawn_subagent` and `SubagentSpec.write_to`
(tuple). Each entry is a workspace-relative file or directory, normalized with `safe_path`.
A spawned sub-agent's grants must lie within what its creator may write (the top agent may grant
any path); refused as an `Error: ` string otherwise. Declared roles' grants come from the host and
are only normalized. Grants are fixed for the sub-agent's life (no change tool: not asked for).

**Instructions.** `subagent_instructions(path, write_to)` states the rule and the grants, and
says it applies to `run_python` too.

**Known issue (documented, not enforced).** `run_python` runs in a sandbox that mounts the whole
workspace read-write, so code a sub-agent runs can write anywhere in the workspace. Enforcing it
needs a per-agent mount (workspace read-only, the agent's folder and grants read-write), and the
local backend cannot enforce it at all. README and CHANGELOG say so.

The overwrite guard stays: it still covers the top agent and paths shared through grants.

## 2. `subagent_middleware`

**API.** `subagent_middleware: Callable[[str], Sequence[middleware]] | None` on
`Session.create_agent`, `Sheath(...)` and module `solve`. Called once per sub-agent, declared
and spawned, with its path; the returned middleware is added after sheath's own
(`_ToolActivity`). Fresh instances per call, so stateful middleware is never shared between
agents running at once. A factory that raises or returns a non-sequence: declared roles raise
at `create_agent`; a spawn is refused with `Error: could not build sub-agent ...`.

**Warning.** `create_agent` with sub-agents or a spawn policy, middleware of the caller's own,
and no `subagent_middleware` warns once (`UserWarning`) that the middleware applies to the
parent only.

## Tasks (TDD)

1. Write confinement: `AgentNode.write_to`, `may_write`, grant normalization and checks in
   `spawn`/declared build, enforcement in the three tools, instructions. Tests in
   `tests/test_write_scope.py`; existing tests that had sub-agents write at the root move their
   writes into the sub-agent's folder.
2. `subagent_middleware`: plumbing, per-sub-agent calls, warning. Tests in
   `tests/test_subagent_middleware.py`.
3. Docs: README (confinement, known issue, middleware), CHANGELOG.
