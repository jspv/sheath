# Roadmap

Status of work beyond the shipped v0.1.1 surface. The README documents what exists;
this file tracks what doesn't yet. Per-item design specs and phased plans live in
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

Last reviewed: 2026-10-02.

## In progress

Nothing currently.

## Delivered

### Workspace trust zones and bounded I/O
Design in `docs/superpowers/specs/2026-10-02-workspace-trust-zones-design.md`. Closed a
container-tier escape (the harness followed links sandboxed code planted in the shared
root) by making handle storage, inputs, scripts and harness state read-only to the sandbox
and moving the exchange into a per-run directory; made the handle store thread-safe;
bounded every built-in tool result and every harness-side read (process output, files,
parquet previews, HTTP bodies); put AG-UI turns under the conversation lock and idle clock;
added CI.

### Security hardening — make the trust boundary real
Delivered 2026-10-01; design in `docs/superpowers/specs/2026-09-17-security-hardening-design.md`.
Zero-trust handle metadata (the parent derives every field from the bytes, at creation and
again on manifest rehydration; handle records are immutable and handle bytes are
digest-verified on read; the control channel is bounded), container backend by default, an egress guard
(`sheath/egress.py`) covering `fetch_url` and `read_document`, and per-conversation tools via
`tool_factory`. Residuals are listed under Planned and Smaller carry-overs.

## Planned

### IP pinning (full DNS-rebinding closure)
The egress guard resolves a host to validate it and the HTTP client resolves it again, so a
DNS entry that changes in between is not caught. Closing it needs connect-to-validated-IP with
a Host override (custom transport, with TLS SNI care). Out of scope for the hardening phase;
the residual TOCTOU window is accepted.

### Micro-VM sandbox tier
gVisor / Firecracker behind the existing `SandboxExecutor` interface, as a third `backend`
value alongside `local` and `container`. Sequenced after the hardening work, which has now made the
container tier the default.

### MAF skills + memory providers (v1.1)
Wire MAF's `Skill`/`SkillsProvider` and `MemoryStore`/`MemoryContextProvider` into the
harness so the agent can accumulate reusable procedures and cross-session memory. Both are
marked experimental upstream and emit `ExperimentalWarning`, so pin behavior carefully.

### Workflow durability / HITL outer shell
Wrap the agent loop in a MAF Workflow to get checkpointing, resume, and human-gated steps —
the durable outer orchestration that the original research reserved Workflows for, with the
single agent loop as a node inside it.

## Smaller carry-overs

Real but low-severity; each is a contained fix.

- **`max_file_size_mb` is silently ignored on the container tier** — now the default backend,
  so the config field does nothing by default. Documented as local-tier-only in the README;
  a container-side equivalent (or a warning when it is set) is still open.
- **`_OrchestratedSandbox` is sequential and not re-entrant per root** — fine at current
  scale; revisit if concurrent runs per conversation are ever needed.
- **Parquet parsing in the host process** — the harness still reads the footer and first
  record batch of a file sandboxed code wrote; a hostile data page is decoded whole. Moving
  description into the sandbox (with the host trusting only sizes) would close it.
- **Fetch deadline gaps** — the total deadline does not bound DNS resolution in
  `validate_url`, and a TCP/TLS connect is bounded only by `FetchConfig.timeout_s`.

## Explicitly out of scope

Decided, not deferred — revisit only with a concrete need.

- Headless-browser / JS rendering for fetches.
- A full search-provider abstraction (Tavily is the one provider).
- Alternative document backends (Docling is the one backend).

## Threat model

Decided 2026-06-15. **The model itself, and any content it pulls from the web or documents,
is treated as potentially adversarial.** Prompt-injection-in-fetched-content escalating to
code execution or SSRF is in scope. Consequences: the sandbox must be a real isolation
boundary, not a nominal one; egress is a model-controlled capability and must be filtered;
and per-conversation resources must not be shared. When touching sandbox, fetch, handles, or
conversation resource wiring, assume the model is an adversary whose capability set is the
tool surface.

Registry scope (in-process / single-event-loop vs. distributed) is **undecided**. Preserve
optionality: invest in the per-conversation lease, the clean async `ConversationStore` seam,
and bounded capacity — all valuable under either answer — and defer persistence and
distribution.
