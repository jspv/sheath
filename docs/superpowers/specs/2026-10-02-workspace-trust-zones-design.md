# Workspace trust zones and bounded I/O — design

Date: 2026-10-02
Status: proposed
Supersedes: the "Control-plane relocation" roadmap entry (previously marked optional).

## 1. Why

A review on 2026-10-02 reproduced the following against the shipped container tier:

1. **Sandbox escape through links in the shared root.** The session root is bind-mounted
   read-write into the container, and the parent keeps trusted state (handle files, the
   manifest, control files, inline scripts) in that same tree. Sandboxed code can plant a
   symlink whose target is a host path; the parent then reads through it (`emit` file →
   host JSON file returned to the model) or writes through it (`handles/h<N>.txt` or
   `handles/_manifest.json.tmp` → host file overwritten with fetched text). Under the threat
   model (the model and fetched content are adversarial) the write is host code execution.
2. **Tools run concurrently; the session assumes they do not.** The agent framework
   dispatches one turn's tool calls on worker threads in parallel. `HandleStore` is not
   thread-safe (92 of 200 concurrent `put()` calls failed on the shared manifest temp name),
   and the file tools resolve a path and then use it without the sandbox lock.
3. **Built-in tools can flood model context.** `run_python` returns uncapped `stdout` (a
   5 MB print became a 10 MB tool result); `inspect_handle` returned a 3 MB preview for a
   one-line text handle.
4. **Untrusted input can exhaust or hang the parent.** Unbounded process-output capture,
   whole-file reads, FIFO opens, whole-row-group parquet previews, whole-dataframe loads in
   the parent, and whole-body HTTP downloads with no total deadline.
5. **The AG-UI path bypasses the conversation lock and the idle clock**, so an active
   thread can be reaped by TTL and two requests on one thread are not serialized.
6. Smaller defects: no CI for tests, racy root allocation, a shared namespace between
   one-shot and conversation roots, a library constructor that mutates process environment
   and the caller's config, an image tag keyed on a Python version the image does not use,
   and documentation drift.

## 2. Goals and non-goals

**Goals**

- G1. No parent-side filesystem operation can be redirected by anything sandboxed code
  wrote. This holds by construction in the container tier, not by per-call-site care.
- G2. Handle bytes and uploaded inputs are immutable to sandboxed code in the container tier.
- G3. Every built-in tool result has a hard size ceiling.
- G4. Every parent-side read of data that sandboxed code or a remote server produced is
  bounded in memory and cannot block indefinitely.
- G5. Parallel tool calls within one turn are safe.
- G6. An AG-UI turn is a conversation turn: serialized, and counted as activity.
- G7. The suite runs in CI on every push and before every publish.

**Non-goals**

- DNS-rebinding closure (IP pinning) and a micro-VM tier stay on the roadmap.
- The local tier remains "not a security boundary". It adopts the same layout and protocol,
  so behavior is identical, but nothing here claims to confine local-tier code.
- A disk quota for the container tier (`max_file_size_mb`) stays a roadmap carry-over.
- No whole-file handle digest. With G2 the existing head-and-tail digest is defence in
  depth for the local tier only; it is kept unchanged.

## 3. Design

### 3.1 Trust zones

One session root, four parent-owned ("protected") directories, and one per-run exchange
directory. Everything else under the root is the model's working area.

| Path under the root | Written by | Container mount |
|---|---|---|
| `handles/` | parent only | `/workspace/handles`, read-only |
| `inputs/` | parent only | `/workspace/inputs`, read-only |
| `.scripts/` | parent only | `/workspace/.scripts`, read-only |
| `.tether/` (`manifest.json`, `tmp/`, `runs/`) | parent only | `/workspace/.tether`, read-only |
| `.tether/runs/<token>/` | parent creates; sandbox writes during one run | `/tether-run`, read-write |
| everything else (`outputs/`, model files) | model and sandbox | `/workspace`, read-write |

Rules:

- **R1.** The four protected directories are created as real directories when the
  workspace is opened, before any sandboxed code runs, and are mounted read-only on every
  run. A mount point cannot be renamed or removed from inside the container, so sandboxed
  code can neither write inside them nor replace them.
- **R2.** Before and after every run the parent `lstat`s the four directories. Anything
  other than a real directory raises `WorkspaceTamperedError`; nothing is ingested. Opening
  a pre-existing root applies the same check. (In the container tier this cannot fire; it
  is the backstop for the local tier and for roots carried over from elsewhere.)
- **R3.** The parent writes only inside protected directories or, for model-requested
  writes, through `safe_path` while holding the session lock (§3.4). The model-facing
  `write_file` refuses any path that resolves into a protected directory.
- **R4.** The parent reads sandbox-written data only from the run directory, only by
  single-segment name, only with `O_NOFOLLOW | O_NONBLOCK`, only if `fstat` says regular
  file, and only up to a stated byte cap.
- **R5.** The manifest moves from `handles/_manifest.json` to `.tether/manifest.json`. A
  legacy manifest found on open is read once (records are untrusted and re-derived, as
  today), rewritten at the new location, and removed.
- **R6.** Scratch downloads for `read_document` move from `<root>/_documents/` to
  `.tether/tmp/`.

### 3.2 Run protocol

Per run the parent creates `.tether/runs/<token>/` (16 hex chars), writes `registry.json`
into it, and passes two environment variables: `TETHER_ROOT` and `TETHER_RUN_DIR`
(`/workspace` and `/tether-run` in the container; host paths in the local tier), plus
`TETHER_PUBLISH_ENABLED=1` when publishing is on. Fixed file names inside the run directory:

| File | Direction | Cap |
|---|---|---|
| `registry.json` | parent → sandbox | — |
| `emit.json` | sandbox → parent | `max_emit_bytes` |
| `new_handles.jsonl` | sandbox → parent | `max_control_bytes` |
| `publish.jsonl` | sandbox → parent | 1 MiB |
| `save_<n><ext>` payload files | sandbox → parent | none (moved, not read into memory) |

`save(id, obj)` writes the payload into the run directory and appends
`{"id", "kind", "file", "source"}`. The parent ingests each record with
`HandleStore.adopt_file`:

- `id` must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` (enforced in the parent; the sandbox
  helper applies the same check to fail early with a clear message);
- `kind` must be `dataframe`, `json` or `text`; the parent chooses the extension and the
  destination `handles/<id><ext>`;
- `file` must be a single path segment; the source is `lstat`-ed and must be a regular file;
- the file is **renamed** into `handles/` (same filesystem, no copy), then described; a
  file that cannot be described is deleted and the record rejected;
- an existing id, or an existing destination file, is refused (`HandleIdReuseError`);
- `source` is truncated to 200 characters.

Rejected records are counted and reported in the run's `error` so the model learns a save
did not take effect. The run directory is removed (`rmtree`, which does not follow links)
when the run ends.

If a timed-out container cannot be removed, the run is marked **orphaned**: the sandbox may
still be running, so nothing is ingested and the result carries an error saying so.

### 3.3 Bounded output and bounded reads

- **Process output.** Both tiers capture `stdout`/`stderr` through a drain that keeps the
  first and last halves of `SandboxConfig.max_output_bytes` (default 8192 per stream) and
  counts what it dropped; the text returned names the number of omitted bytes. Parent
  memory per stream is bounded by the cap.
- **`run_python` result.** An emitted result whose JSON encoding exceeds
  `spill_threshold_bytes` is stored as a handle and replaced by
  `{"spilled": true, "handle": <summary>}`.
- **Handle summaries.** A dataframe preview is produced from the first record batch
  (`iter_batches(batch_size=5)`), not the first row group; every preview is capped at 800
  characters; `schema` lists at most 200 columns with names cut to 128 characters.
- **`inspect_handle`.** `rows` is clamped to 1..100; the returned preview or head is capped
  at 20 000 characters; text and JSON previews are read from the file with a bounded read
  (JSON is shown as stored, not re-indented); `stats=True` returns per-column
  `min`/`max`/`null_count` from parquet footer statistics. The parent never materializes a
  dataframe handle.
- **`publish_handle`.** Dataframe conversion (csv/xlsx/parquet/json) runs inside the sandbox
  via a fixed parent-authored script that writes `outputs/<name>`; the parent then publishes
  that file through the existing no-follow validation.
- **File tools.** `read_file` requires a regular file, rejects binary content (NUL in the
  first 8 KiB) with a clear message, and reads line-by-line with a per-line bound;
  `list_files` skips links and `.tether/` and lists at most 2000 entries; `search` requires
  a directory or regular file, skips links and files over 50 MiB, clamps `max_matches` to
  1000, bounds the ripgrep subprocess with a 30 s timeout and streamed output, and trims
  the returned hits to a 60 000-character budget with a note saying how many were omitted.
- **HTTP.** `guarded_get` is replaced by `guarded_fetch`, which streams, stops at the byte
  limit, enforces `FetchConfig.total_timeout_s` (default 120 s) across all hops, and checks
  every hop's scheme against `allowed_schemes`. The model-supplied `max_bytes` is clamped to
  `FetchConfig.max_bytes_ceiling` (default 100 MB). `read_document` refuses a truncated
  download instead of converting a partial file.

### 3.4 Concurrency

- `HandleStore` gets an internal re-entrant lock around every mutation, unique temp names,
  and atomic temp-plus-rename writes for handle files and the manifest.
- `read_file`, `write_file`, `list_files`, `search`, the local-path branch of
  `read_document`, and both publish tools hold the session's `io_lock` (the sandbox run
  lock), so no sandboxed code runs between a path check and its use.
- `read_document` on a workspace path copies the file to `.tether/tmp/` under the lock and
  converts the private copy outside it.
- Network tools (`fetch_url`, `web_search`, `web_extract`) do not take `io_lock`; they only
  touch the store, which has its own lock. Lock order is always `io_lock` then store lock.

### 3.5 Conversations

- `Conversation.astream_agui(input_data, **kwargs)` runs the AG-UI stream under the
  conversation lock and stamps `last_activity` at start and end. `Tether.agui_stream`
  delegates to it.
- `Conversation.busy` is true while a turn holds the lock. The manager never treats a busy
  conversation as expired, and re-opening a live id counts as activity.
- Conversation ids are validated with `validate_segment`.
- One-shot roots are allocated with `mkdtemp` under `./.tether/runs/`; conversation roots
  stay under `./.tether/sessions/<id>`. The two namespaces no longer overlap, and
  allocation is atomic across threads and processes.

### 3.6 Packaging and hygiene

- `Tether.__init__` no longer calls `load_dotenv()` and no longer mutates the caller's
  config. The Tavily key is read from the environment, then from `.env` via
  `dotenv_values`, into a copied config. `python-dotenv` becomes a declared dependency.
- The sandbox Containerfile takes `PYTHON_VERSION` as a build argument, set to the host's
  `major.minor`, so the image tag and the pip-layer key describe the interpreter that runs.
- A `tests` workflow runs lint and the suite on push and pull request; `publish` requires it.
- README, CHANGELOG and ROADMAP are corrected (default root location, digest coverage,
  layout, new config fields, the superseded roadmap entry).

## 4. Compatibility

- **Model-visible:** scripts that write directly into `handles/`, `inputs/` or `.scripts/`
  now fail with a read-only error in the container tier (they were never registered as
  handles). `save()` rejects ids outside the pattern in §3.2. `inspect_handle(stats=True)`
  returns footer statistics, not `describe()`. JSON previews in `inspect_handle` are not
  re-indented.
- **Host-visible:** `egress.guarded_get` is removed in favor of `guarded_fetch`.
  One-shot session directories are named `./.tether/runs/run-XXXXXXXX`, not
  `./.tether/sessions/<N>`. `Tether(config).config` may be a copy of the object passed in.
  `OPENAI_API_KEY` and other `.env` values are no longer exported into `os.environ` by
  constructing a `Tether`.
- **On-disk:** existing roots are migrated on open (R5). Stray control files from the old
  protocol are ignored.
- New config: `SandboxConfig.max_output_bytes`, `FetchConfig.total_timeout_s`,
  `FetchConfig.max_bytes_ceiling`.

## 5. Testing

Every unit is test-first. In addition to unit tests per component:

- **Attack regressions (local tier, always run):** a symlink at `emit.json`,
  `new_handles.jsonl`, a payload file, and `handles/` itself each yield an error or
  `WorkspaceTamperedError`, and a sentinel file outside the root is neither read nor
  modified.
- **Attack regressions (container tier, gated on a live runtime):** writing, symlinking,
  renaming or removing anything under `handles/`, `inputs/`, `.scripts/`, `.tether/` fails
  inside the sandbox; the three reproduced attacks leave the sentinel untouched.
- **Concurrency:** 4 threads × 50 `put()` calls produce 200 distinct handles and no errors.
- **Ceilings:** a parametrized test asserts each built-in tool's serialized result stays
  under its ceiling for a hostile input.

## 6. Residual risks (to be stated in the README)

- The parent still parses a parquet footer and first record batch written by sandboxed
  code, and an oversized data page is decoded whole.
- The local tier provides none of the guarantees above.
- Container-tier disk use is unbounded.
- DNS rebinding between validation and connect is unchanged.
