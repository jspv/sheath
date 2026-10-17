# sheath × CopilotKit (with an MCP server)

A full end-to-end example: a [CopilotKit](https://docs.copilotkit.ai/) chat UI in the
browser, talking to the sheath over [AG-UI](https://docs.ag-ui.com/), with an **MCP
server** wired into the agent. Ask a data question in the chat and watch the agent call
the MCP tool, spill the large result to a handle, and write sandboxed Python to compute
the answer — with live progress streamed back to the UI.

## Architecture

```
Browser (CopilotKit UI)
  │  POST /api/copilotkit            (CopilotKit protocol, same-origin)
  ▼
Next.js runtime route               app/api/copilotkit/route.ts
  │  AG-UI RunAgentInput over HTTP/SSE   (HttpAgent → the Python backend)
  ▼
FastAPI  POST /agent                backend/server.py
  │  sheath.agui_stream(input_data, tools=[sales_mcp])
  ▼
sheath agent  ──(stdio MCP)──▶     backend/mcp_server.py   (sales_rows tool)
  │
  └─ spills the big MCP result to a handle, runs sandboxed Python, streams events back
```

The runtime route proxies on the server, so **no CORS** setup is needed on the Python
side. CopilotKit generates and forwards a `threadId` per conversation; the sheath maps
it to a persistent workspace, so handles and files persist across turns in a thread.

## Prerequisites

- Python 3.12 + [`uv`](https://docs.astral.sh/uv/), with the `agui` extra installed
- Node.js 18+ (for the Next.js frontend)
- An `OPENAI_API_KEY` in your environment or `.env` (the sheath's default client — see
  the repo README's *Bring your own model client* to use Azure/Foundry/etc.)

## 1. Run the backend

From the repository root:

```bash
uv sync --prerelease=allow --extra agui
# Pass --extra agui to `uv run` too: a bare `uv run` re-syncs to the default deps and
# would drop the agui extra (ag_ui / fastapi), breaking the import.
uv run --prerelease=allow --extra agui \
  uvicorn examples.copilotkit.backend.server:app --reload --port 8000
```

This serves the AG-UI endpoint at `http://localhost:8000/agent`. The `sales` MCP server
is launched on demand (stdio) for each new conversation — you don't start it yourself.

## 2. Run the frontend

In a second terminal:

```bash
cd examples/copilotkit/frontend
npm install        # or: npm install @copilotkit/react-core @copilotkit/react-ui @copilotkit/runtime @ag-ui/client
npm run dev
```

Open <http://localhost:3000>. If your backend runs elsewhere, set
`SHEATH_AGENT_URL` for the frontend (e.g. `SHEATH_AGENT_URL=http://host:8000/agent npm run dev`).

## 3. Try it

In the chat sidebar, ask something the `sales` MCP server can answer:

- *“What was total EU revenue in 2025, excluding invalid rows?”*
- *“Compare EU vs NA revenue for 2025 and show the gap.”*
- *“Which region had the most invalid rows?”*

The agent calls `sales_rows(region)`, which returns a full year of daily rows. That
return is large enough to **spill to a typed handle** rather than entering the model's
context; the agent then writes `run_python` to load the handle, filter invalid rows, and
sum revenue. Tool-call and progress events stream into the chat as it works.

## How the MCP server is attached (and why per request)

`backend/server.py` builds a **fresh** `MCPStdioTool` for each HTTP request and passes it
to `agui_stream(input_data, tools=[...])`:

```python
def _sales_mcp() -> MCPStdioTool:
    return MCPStdioTool(name="sales", command=sys.executable, args=[str(_MCP_SERVER)])
```

`agui_stream` opens one persistent `Conversation` per `threadId`, and that conversation
**connects the MCP server and owns its lifecycle** (closing it on teardown). A new
thread's first message connects this instance; reused threads return the existing
conversation and ignore the fresh (unconnected) object, which is just garbage-collected.
Sharing one connected instance across threads would let the first thread to finish close
the server out from under the others — so per-request construction is the correct seam.

## Files

```
backend/
  server.py       FastAPI AG-UI endpoint → sheath.agui_stream (attaches the MCP per request)
  mcp_server.py   self-contained stdio MCP server: sales_rows(region) demo tool
frontend/
  app/api/copilotkit/route.ts   CopilotKit runtime route → AG-UI HttpAgent
  app/providers.tsx             <CopilotKit …> + the Safari fetch-bind shim (see Troubleshooting)
  app/layout.tsx                renders <Providers>
  app/page.tsx                  the page + <CopilotSidebar>
  package.json, tsconfig.json, next.config.mjs
```

## Troubleshooting

**`agent_connect_failed` — "Can only call Window.fetch on instances of Window".**
This is Safari/WebKit-specific. CopilotKit runs the AG-UI `HttpAgent` connection **in the
browser**, and `@ag-ui/client` calls a *detached* global `fetch` (`this.fetch(...)`), which
Safari rejects (Chrome tolerates it — the classic "works in Chrome, breaks in Safari"). The
fix ships in `app/providers.tsx`: a one-time `window.fetch = window.fetch.bind(window)` that
runs before CopilotKit mounts. If you remove that shim, use Chrome.

**`INCOMPLETE_STREAM` / "terminated" / `No tool output found for function call …` on a turn
that follows a tool call.** Cause: CopilotKit replays the conversation each turn but emits each
backend tool *result* message **before** the assistant message that declared the calls, while
the OpenAI API requires the assistant `tool_calls` message first, then its results — so a
`function_call` looks like it has no output and the request 400s. The sheath fixes this in
core: `agui_stream` reorders the replayed messages (`sheath.agui.repair_tool_message_order`)
before running the agent, so **no action is needed** here. (If you build your own AG-UI backend
without going through `agui_stream`, you must reorder the messages yourself.)

**Server restarts mid-conversation.** `uvicorn --reload` watches the repo, and the agent's
`run_python` writes sandbox scripts into the session dir. If that dir is under the repo (the
sheath default `./.sheath/...`), the write hot-reloads the server **mid-conversation** —
aborting the in-flight response and wiping the in-memory session registry. This example keeps
the session root in a temp dir (`server.py` sets `SheathConfig(root_dir=…)`), so sandbox
writes never trigger a reload. More generally, for continuous threads, avoid losing the live
conversation mid-thread (don't hot-reload; leave `idle_ttl_s` unset or generous).

> **Versions:** `package.json` pins a known-good snapshot — `@copilotkit/* ^1.60` with
> `@ag-ui/client 0.0.56` (the exact version `@copilotkit/runtime@1.60` depends on; keep
> these aligned, or a build-time `HttpAgent`-type mismatch can appear). CopilotKit and AG-UI
> move quickly; if `npm install` resolves something incompatible, check the
> [CopilotKit AG-UI docs](https://docs.copilotkit.ai/) for wiring changes and re-align the
> `@ag-ui/client` version with whatever `@copilotkit/runtime` pulls in.
