# Changelog

0.2.0 is a breaking release: the project was renamed, the default model and several defaults
changed, and the minimum framework versions rose. Entries marked **Breaking** or
**Behavior change** need action when upgrading from `tether-harness` 0.1.x.

## 0.2.1 — 2026-10-21

### `load` / `save` work inside `run_python` without an import
- **Fix:** the code bundle tells the model that `load(id)` and `save(id, obj)` need no import,
  but the script ran in a bare namespace, so `load('h1')` raised `NameError`. The runner now
  provides `load`, `save` and `emit`, plus `publish` when the host accepts files, on both
  sandbox tiers.

### MCP progressive disclosure keeps the spill guarantee
- **Fix:** with MAF's `use_progressive_disclosure=True`, a tool the model loaded mid-run with
  `load_tool` never got the spill parser or a progress token, so its full result went straight
  into the conversation. Every allowed tool is now wired when the server connects. If a future
  agent-framework version removes the seam this relies on, connecting fails instead of letting
  deferred tools skip the spill.

### `agent_kwargs` on the facade paths
- `Sheath(..., agent_kwargs={...})` sets framework agent options (for example
  `skills_provider`, `max_output_tokens`) for every conversation; `aopen`, `asolve` and `solve`
  take a per-call `agent_kwargs` that wins key by key, with `default_options` merged one level
  down. `agui_stream` uses the `Sheath` default.
- **Behavior change:** `Session.create_agent` (and so `agent_kwargs`) now refuses
  `shell_executor`, `file_access_store` and `background_agents` with `TypeError`. Each would give
  the model a shell, files or other agents outside the sandbox, `safe_path` and egress checks.

## 0.2.0 — 2026-10-16

### Conversation history stays local (`store=False`)
- **Behavior change (privacy):** the built-in OpenAI client stores conversations at OpenAI by
  default. Agents from `Session.create_agent()` now send `store=False`, so the web content, documents
  and workspace data they handle are not retained server-side and history is kept in-process.
  Opt back in per agent with `default_options={"store": True}`; other `default_options` you pass are
  preserved. Multi-turn memory is unaffected (covered by a live test), and the framework's
  "stores history server-side" warning no longer appears.

### Default model is now `gpt-6.1-sol`
- **Behavior change:** the built-in OpenAI client, the CLI `--model` flag and `SheathConfig.model`
  default to `gpt-6.1-sol` (previously `gpt-5-mini`). It costs more per token; pass
  `--model gpt-5-mini` or `SheathConfig(model="gpt-5-mini")` to keep the old model, which your
  account may still serve. Chosen from a 10-task eval against `gpt-5-mini`, `gpt-5.4-mini` and
  `gpt-6-luna`: all answered the checkable tasks correctly, but `gpt-6-luna` stated data-quality
  checks in 0 of 10 answers where `gpt-6.1-sol` did in 7 of 10.
- `evals/run_evals.py` was ported to the current conversation API (it had been calling removed
  `solve(on_tool_call=...)` and `Sheath.session`), and takes `SHEATH_EVAL_MODEL` / `SHEATH_EVAL_OUT`.

### Dependency upgrades (minimum versions raised)
- **Breaking for installs:** requires `agent-framework-core>=1.22.0` and
  `agent-framework-openai>=1.17.0` (previously `>=1.8.1`); the `agui` extra requires
  `agent-framework-ag-ui>=1.6.0` (previously `>=1.0.0rc4`). The `ag-ui` package cannot be
  upgraded without the core upgrade, so the three move together.
- The framework renamed `create_harness_agent(disable_memory=...)` to `disable_file_memory`, and its
  default is now *on* (a file-backed memory rooted at `{cwd}/agent-file-memory`); we pass
  `disable_file_memory=True` so the agent gets no host-filesystem memory tool.
- The framework now wires a tool-approval middleware by default that raises unless every
  `run()` is given an `AgentSession`; we disable it (overridable through `maf_kwargs`), so
  `Session.create_agent()` agents can still be run without a session.
- `docling` locked at 2.140.0 (was 2.100.0), with `docling-parse` 7 and `docling-ibm-models` 4.

### Renamed to `sheath` (breaking)
- **Breaking:** the project is now `sheath`. The PyPI distribution is `sheath` (previously
  `tether-harness`) and the import package is `sheath` (previously `tether`). Install with
  `pip install sheath` and `from sheath import Sheath, SheathConfig, solve`.
- **Breaking:** `Tether` / `TetherConfig` are now `Sheath` / `SheathConfig`; `Session.tether_instructions`
  is now `Session.sheath_instructions`.
- **Breaking:** environment variables use the `SHEATH_` prefix (`SHEATH_ROOT`,
  `SHEATH_SANDBOX_BACKEND`, `SHEATH_LIVE`, ...); `TETHER_*` is no longer read.
- **Breaking:** console scripts are `sheath`, `sheath-prefetch-docling` and
  `sheath-build-sandbox`.
- **Breaking:** the on-disk state directory is `.sheath/` and the in-container exchange mount is
  `/sheath-run`. Existing `.tether/` directories are not migrated. Sandbox images are tagged
  `sheath-sandbox:*`, so the first run builds (or re-tags) the image once.

### Workspace trust zones and bounded I/O
- **Security fix:** sandboxed code could plant a symlink in the session root that the
  harness then read or wrote through, reaching host files outside the root from the
  container tier. `handles/`, `inputs/`, `.scripts/` and the new `.sheath/` are now mounted
  read-only into the container; sandbox-to-harness data travels only through a per-run
  exchange directory read with no-follow, size-capped opens; saved payloads are moved into
  `handles/` by the harness. A replaced protected directory raises `WorkspaceTamperedError`.
- **Behavior change:** scripts can no longer write into `handles/`, `inputs/` or `.scripts/`
  directly (container tier); use `save(id, obj)`. `save()` ids must match
  `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. `write_file` refuses those directories.
- **Behavior change:** the manifest moved to `.sheath/manifest.json` (an existing
  `handles/_manifest.json` is migrated on open). `read_document` scratch files moved to
  `.sheath/tmp/`.
- **Behavior change:** one-shot roots are `./.sheath/runs/run-XXXXXXXX` (atomic allocation),
  no longer `./.sheath/sessions/<N>`.
- `HandleStore` is thread-safe and writes atomically; parallel tool calls in one turn no
  longer fail on the manifest.
- File tools hold the session lock, read only regular files, and are bounded; `read_file`
  rejects binary files with a clear error.
- `run_python` output is capped (`SandboxConfig.max_output_bytes`, default 8192 per stream)
  and a large emitted result is stored as a handle (the tool returns
  `{"spilled": true, "note": ..., "handle": {...}}`). When a run ends, normally or by
  timeout, the local tier kills the child's whole process group.
- `inspect_handle` is capped, clamps `rows` to 100, reads text/JSON previews raw and
  bounded (JSON is no longer pretty-printed), and `stats=True` now returns footer
  min/max/null_count instead of `describe()`; min/max are null when any non-empty row group
  lacks them. A first row group over 16 MiB yields no head rows (`head_omitted`), and the
  same guard omits the handle-summary preview. `publish_handle` converts dataframes inside
  the sandbox.
- **API change:** `egress.guarded_get` is replaced by `egress.guarded_fetch`, which streams
  with a byte cap, enforces `FetchConfig.total_timeout_s` (120 s), checks each hop's scheme,
  and returns a `FetchedBody`. The deadline is enforced by a watchdog that shuts down the
  connections opened during the call and raises `FetchDeadlineExceeded` (an
  `httpx.HTTPError`); it does not bound DNS resolution in `validate_url`, and the TCP/TLS
  connect is bounded only by the client's per-phase timeout (`FetchConfig.timeout_s`). New
  `FetchConfig.max_bytes_ceiling` (100 MB) clamps a model-supplied `max_bytes`.
- `read_document` refuses (rather than truncates) a URL download larger than
  `FetchConfig.max_bytes`, and caps a workspace path at `SheathConfig.max_input_bytes`
  while copying it privately.
- AG-UI turns take the conversation lock and count as activity, so `idle_ttl_s` no longer
  reaps an active thread (a stream a server never closes keeps the conversation exempt);
  conversation ids are validated as a single safe path segment (`aopen("")` is rejected).
  Sync `Conversation.add_input` raises while an AG-UI stream or `aask` turn is open; use
  `aadd_input`.
- **Behavior change:** `Sheath(...)` no longer calls `load_dotenv()` or mutates the config
  passed in. `TAVILY_API_KEY` is read from the environment, then from `./.env` in the
  current working directory; other `.env` values (such as `OPENAI_API_KEY`) are no longer
  exported into `os.environ` -- the built-in OpenAI client reads `.env` itself.
  `python-dotenv` is a declared dependency.
- The sandbox image is built for the host's Python `major.minor`.
- **Security fix:** `publish_handle` followed a link sandboxed code planted at `outputs/`
  (for example `outputs -> handles`), so the harness could overwrite handle files, the
  manifest or an upload. A link or file at `outputs/` is now replaced by a real directory
  before anything is written.
- **Security fix:** `.sheath/tmp/` and `.sheath/runs/` are created with the workspace and
  must be real directories (`WorkspaceTamperedError` otherwise); scratch and run directories
  are never created through a link.
- **Security fix:** a timed-out container that the runtime cannot confirm removed
  quarantines the session: later runs and file tools retry removing that exact container
  and raise `SandboxQuarantinedError` (new, exported) until it is gone. A container that
  had already exited is no longer misreported as orphaned.
- **Security fix:** an unreachable container runtime (e.g. a missing socket, whose error
  reads "no such file or directory") no longer counts as "container gone": the timed-out
  run is reported orphaned and an existing quarantine stays until the runtime itself says
  `no such container`.
- **Security fix:** the Python `search` fallback runs in a child process killed after 30 s,
  so a catastrophic regex no longer freezes the host process.
- `Conversation.aclose` (and `SessionManager.close`/`aclose`) waits up to 30 s for a running
  turn before tearing the workspace down; turns that start after the close see the
  conversation closed (`aask` returns an error, `astream_agui`/`aadd_input` raise).
- `SessionManager.aopen` reaps an expired conversation outside the manager lock, so a slow
  teardown no longer stalls opens of other ids; opens of the same id (and opens racing a
  `sweep`/`close` of it) wait until the old workspace is removed, so the replacement's root
  is never deleted from under it.
- CI runs lint and tests on push and pull request; publishing requires them. The
  container-tier tests run in CI when a runtime is usable (ubuntu-latest has one) and
  build the sandbox image there.

### Isolation is on by default
- **Behavior change:** `SandboxConfig.backend` now defaults to `"container"`, not `"local"`.
  `run_python` executes model-authored code, so the shipped default is a real boundary.
  `SHEATH_SANDBOX_BACKEND` overrides the default and selects the *tier* (`container` or
  `local`), not the container runtime; an explicit `SandboxConfig(backend=...)` still wins.
  Any other value raises `ValueError` rather than falling back.
- Selecting `local` emits `NoSandboxIsolationWarning`.
- Selecting `container` probes the runtime for *liveness* (`<runtime> info`), not just
  presence on `PATH`, and raises `SandboxRuntimeUnavailable` when it is absent or
  unresponsive. Successful probes are cached per process.

### Zero-trust handle metadata
- Sandboxed code reports only `{id, kind, path, source}` for a file it writes; the parent
  derives every described field (`preview`, `bytes`, `schema`, `n_rows`, `n_cols`) from the
  bytes on disk, including on manifest rehydration.
- Handle records are immutable: reusing an id is refused (`HandleIdReuseError`) and reported
  back to the model.
- A content digest is recorded when metadata is derived and checked on every read; a
  mismatch raises `HandleTamperedError`. The digest covers the file's length, its first 64 KiB and its last 64 KiB.
- New `SheathConfig.max_emit_bytes` (1 MiB), `max_control_bytes` (8 MiB) and
  `max_new_handles` (256) bound the parent↔sandbox control channel.

### Egress policy
- New `sheath/egress.py`. `fetch_url` and `read_document` deny loopback, private,
  link-local, reserved, multicast, unspecified and CGNAT (`100.64.0.0/10`) addresses by
  default, and re-validate every redirect hop (`follow_redirects=False` is forced).
- Cloud metadata endpoints are denied unconditionally and cannot be allowlisted.
- New `FetchConfig.allow_private_hosts` (hostnames or CIDRs) and `FetchConfig.max_redirects`.

### Per-conversation tools
- New `tool_factory(build)` marker: each conversation builds its own instance of a stateful
  tool (such as an MCP connection) instead of sharing one.

### Isolation can be made mandatory
- New `SandboxConfig.require_isolation: bool = False`. With `container` as the default this
  is the stricter assertion on top of it: when true, a non-container backend is refused
  *however it was selected* -- an explicit `backend="local"` or `SHEATH_SANDBOX_BACKEND=local`
  -- raising `SandboxRuntimeUnavailable` at `Session.create`; a missing container runtime or
  image raises and never falls back to the local tier.
- A missing container runtime now raises `SandboxRuntimeUnavailable` (a `RuntimeError`
  subclass) at `Session.create`.
- Documented that the local tier is **not a security boundary**.

### Sandbox image readiness
- New `SandboxConfig.build_on_demand: bool | None = None` (`None` → build unless
  `require_isolation`). With builds disabled, a missing image or pip layer raises
  `SandboxImageMissing` immediately with the command that builds it.
- New `SandboxConfig.build_timeout_s: float = 900.0` bounds image builds and pip-layer
  provisioning (`SandboxImageError` on expiry). `image inspect` is bounded to 10 s.
- New `sheath.sandbox_preflight(config) -> PreflightReport(ok, backend, runtime, image_tag,
  problems)`: a non-mutating readiness check that never builds.
- `sheath-build-sandbox` accepts `--preinstalled`, `--pip`, `--runtime`, `--timeout`, and
  `--check`.

### Agent instructions
- `agent_instructions` on `Sheath(...)` (default), `Sheath.aopen` / `SessionManager.aopen`
  (per-conversation override), `asolve` / `solve`, and `Conversation.acreate`.

### Publishing files to the host
- New opt-in `deliver` bundle: `publish_file(path, name=None, description=None)` and
  `publish_handle(handle_id, format=None, name=None, description=None)`; exposed only when a
  host callback is configured.
- New `publish(path, name=None, description=None)` helper inside `run_python`; requests are
  processed after a clean exit and reported in `ExecResult.published`.
- New `Sheath(on_publish=...)` / `aopen(on_publish=...)` receiving a `PublishedFile`.
- New `SheathConfig.max_publish_bytes: int = 100 MB`.
- The callback receives a private snapshot of the validated bytes (outside the workspace),
  deleted when it returns.

### Sandbox run safety
- Sandbox runs are serialized per session, and publishing / input ingestion take the same
  lock, so no sandboxed code runs while the parent touches paths in the workspace.
- A container that exceeds `timeout_s` is now force-removed by name; previously only the
  runtime CLI was killed and the container kept running.
- Control files use random per-run names.

### Known-gap fixes
- Publishing never follows links, so its answers no longer reveal whether a host path exists.
  **Behavior change:** publishing through a symlinked directory inside the workspace is
  refused; use the real path.
- `add_input` re-copies an input whose recorded file is missing, replaced, or forged (same
  handle id), repairs entries planted under `inputs/`, and bounds host-path copies. Sandboxed
  code can no longer create input records.
- **Behavior change:** a repeat `add_input` with the same `input_id` but different bytes now
  stores the host's new bytes (same handle id); with the same bytes it is still a no-copy
  no-op.
- `aadd_input` copies in a worker thread.
- Sandbox `publish()` requires the `deliver` bundle (`Session.create(..., bundles=None)` keeps
  the previous behavior for direct callers).
- `build_timeout_s` is one budget for the image build and pip layer; a timed-out pip-layer
  container is removed.

### Ingesting user files
- New `Conversation.add_input` / `aadd_input` / `inputs` (and `Session.add_input` /
  `Session.inputs`): uploads become read-only `binary` handles under `inputs/<input_id>/`,
  idempotent by `input_id` across restarts, never parsed in the host process.
- New `Handle` fields `input_id`, `content_type`, `description` (omitted when unset).
  On rehydration only `input_id` is carried from the (sandbox-writable) manifest;
  `content_type` is re-derived from the filename and `description` is dropped.
- New `SheathConfig.max_input_bytes: int = 100 MB`.
- The container tier mounts `inputs/` read-only.
