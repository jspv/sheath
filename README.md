# sheath

> A reusable Python substrate for building autonomous data-gathering and data-integration agents — a single-agent-loop ("coding agent") harness on the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (MAF).

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Built on Microsoft Agent Framework](https://img.shields.io/badge/Built%20on-Microsoft%20Agent%20Framework-512BD4)

You give it a task and a set of tools (or MCP servers). It fetches data from links and tools, keeps large datasets **out of the model's context** as typed *handles*, and uses a **sandboxed Python environment** to write and run code that analyzes that data — returning a trustworthy answer plus a re-openable audit trail of every script and artifact.

This is **v1: the general substrate**. Specific problems plug in later as user-supplied tools / MCP servers.

## Key features

- **References, not payloads.** Large or structured returns from tools, MCP servers, and URLs spill to typed **handles** (JSON / text / Parquet) *before* they reach the model — ~200 tokens stand in for megabytes.
- **A computer for the agent.** A sandboxed `run_python` speaks the same handle store (`load(id)` / `save(id, obj)`), so the agent writes and runs code to do the real work; only what it deliberately returns flows back into context.
- **MAF-native.** Builds a standard MAF agent (`create_harness_agent`) you can run, stream, compose into workflows, or use as a sub-agent — with any MAF chat client (OpenAI by default; Azure, Foundry, or your own).
- **One-shot and continuous.** Ephemeral, self-cleaning `solve()` for one-shots; a persistent multi-turn `Conversation` (workspace + history) with an optional idle TTL for continuous use.
- **Tools and MCP, auto-handled.** Plain Python callables and MCP servers (stdio / HTTP / WebSocket) drop into one `tools=[…]` list; spill handling and server lifecycle are managed for you.
- **Live status.** Built-in, developer (`report_progress`), and MCP-server progress all stream through one `on_status` feed, `--verbose`, or AG-UI events.
- **AG-UI / CopilotKit.** Stream answer text, tool-call visibility, and status as AG-UI events; frontend tools, shared state, human-in-the-loop, and multi-turn history all work.
- **Confined by construction.** One session root holds everything; every model-supplied path passes through a single `safe_path` chokepoint, and the container sandbox tier (real isolation) is the default, and parent-side fetches are filtered against internal address space.

## Table of contents

- [How it works](#how-it-works)
- [Getting started](#getting-started)
  - [Installation](#installation)
  - [API keys](#api-keys)
  - [Quickstart: CLI](#quickstart-cli)
  - [Quickstart: library](#quickstart-library)
- [Examples and samples](#examples-and-samples)
- [Guides](#guides)
  - [Use it as a MAF agent](#use-it-as-a-maf-agent)
  - [Bring your own model client](#bring-your-own-model-client)
  - [Sessions: one-shot vs continuous](#sessions-one-shot-vs-continuous)
  - [Exchanging files with the user](#exchanging-files-with-the-user)
  - [MCP servers](#mcp-servers)
  - [Live status updates](#live-status-updates)
  - [AG-UI and CopilotKit](#ag-ui-and-copilotkit)
- [Reference](#reference)
  - [Tool surface](#tool-surface)
  - [Security and confinement](#security-and-confinement)
  - [Sandbox tiers](#sandbox-tiers)
  - [Configuration](#configuration)
- [Development](#development)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)

## How it works

The central design move is keeping large data out of context. When a tool, MCP server, or URL returns something big or structured, a **"spill" middleware** intercepts it *before it reaches the model* and writes it to disk as a typed **handle** (JSON, text, or a dataframe stored as Parquet). The model only ever sees a compact summary:

```json
{ "handle": "h1", "kind": "dataframe", "source": "query_sales(region='EU')",
  "schema": {"date": "date", "units": "int", "revenue": "float"},
  "n_rows": 480000, "n_cols": 3, "bytes": 18402211,
  "path": "handles/h1.parquet",
  "preview": "date,units,revenue\n2025-01-01,12,840.0\n... (5 of 480000 rows)" }
```

~200 tokens stand in for 18 MB. To actually work with the data, the agent is given a **computer**: it writes `.py` scripts and runs them in a sandbox that speaks the same handle store (`load(id)` / `save(id, obj)`). Only the small value it deliberately prints/returns flows back into context; derived data becomes new handles. The loop follows a **search → read → analyze** triad: `search` to locate, `read_file` to load the right slice, `run_python` to analyze.

## Getting started

### Installation

Requires **Python 3.12**. Install from PyPI:

```bash
pip install sheath                # or: uv add sheath
pip install "sheath[agui]"        # + AG-UI / CopilotKit streaming
pip install "sheath[docling]"     # + document ingestion (read_document)
```

To work on sheath itself, clone the repository and use [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Built on Microsoft Agent Framework `agent-framework-core>=1.22.0` and `agent-framework-openai>=1.17.0` (and `agent-framework-ag-ui>=1.6.0` for the AG-UI extra); these are the versions the test suite and live tests run against. The lock includes a few prerelease transitive dependencies, so `pyproject.toml` sets `prerelease = "allow"` for `uv` — no flag is needed.

In a checkout, the optional extras are:

```bash
uv sync --extra docling   # document ingestion (read_document)
uv sync --extra agui      # AG-UI / CopilotKit streaming
```

`uv sync` makes the environment match exactly what you ask for, so it **uninstalls an extra you leave off**. Name every extra you want each time, e.g. `uv sync --extra agui --extra docling`.

Document ingestion uses [Docling](https://github.com/DS4SD/docling), which downloads its models on first use. To pay that cost up front (e.g. at deploy time) instead of on the first `read_document` call, prefetch them:

```bash
sheath-prefetch-docling          # add --ocr to also fetch OCR models (scanned documents)
```

OCR is **off by default** — born-digital PDFs/Office files get their tables and structure without it. Set `SheathConfig.documents.ocr = True` for scanned/image documents.

### API keys

Create a `.env` in the project root:

```bash
OPENAI_API_KEY=sk-...        # required by the default OpenAI client (see "Bring your own model client" to use another provider)
TAVILY_API_KEY=tvly-...      # optional — enables web_search / web_extract
```

`TAVILY_API_KEY` is read from the environment, then from `./.env` in the current working directory. Constructing a `Sheath` does not export `.env` into the process environment; the built-in OpenAI client reads `.env` itself.

### Quickstart: CLI

```bash
uv run sheath "What is the total EU revenue in 2025, excluding invalid rows?"
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt-6.1-sol` | Model for the built-in OpenAI client (reads `OPENAI_API_KEY`, or `AZURE_OPENAI_*` for Azure routing). The CLI uses that client; for other providers drive the sheath from Python with your own client — see [Bring your own model client](#bring-your-own-model-client) |
| `--root` | a fresh session dir | Workspace root (the confinement boundary) |
| `-v`, `--verbose` | off | Print live tool status to stderr as the task runs (see [Live status updates](#live-status-updates)) |

The CLI prints the answer and leaves `[session: …]` — the directory holding every script, handle, and artifact for inspection.

### Quickstart: library

```python
from sheath import Sheath, SheathConfig, solve

# One-shot convenience
result = solve("Summarize the latest figures from https://example.com/report.pdf")
print(result.final_text)

# Reusable sheath with your own tools
def query_sales(region: str) -> list[dict]:
    """Return sales rows for a region."""
    ...

# With no client passed, the sheath builds the default OpenAI client (reads OPENAI_API_KEY).
# `model` configures ONLY that default client — to use Azure / Foundry / any other provider,
# pass your own MAF chat client (see "Bring your own model client"); `model` is then ignored.
h = Sheath(SheathConfig(model="gpt-6.1-sol"))
result = h.solve(
    "Total EU revenue in 2025, excluding invalid rows?",
    tools=[query_sales],          # plain Python callables; auto-wrapped + spill-handled
)

result.final_text   # the answer
result.handles      # dict of handle summaries produced during the run
result.files        # user-meaningful files written under the session root
result.session_dir  # full audit trail (scripts, handles, transcript)
result.error        # None, or a string if the run failed (e.g. context overflow)
```

Your tools are wrapped as MAF agent tools automatically, and their returns pass through the same spill middleware — so tool-produced and code-produced data are identical kinds of handle.

## Examples and samples

The quickstarts above and the guides below are each worked examples. Two runnable samples ship in the repo:

| Sample | Shows |
|---|---|
| [`examples/copilotkit/`](https://github.com/jspv/sheath/tree/main/examples/copilotkit) | **Full end-to-end app:** a CopilotKit chat UI + a FastAPI AG-UI backend with an **MCP server** wired in — ask a data question and watch the agent call the MCP tool, spill the result to a handle, and run sandboxed Python to answer |
| [`examples/agui_server.py`](https://github.com/jspv/sheath/blob/main/examples/agui_server.py) | The minimal AG-UI backend: a FastAPI endpoint that streams a sheath run as SSE |
| [Use it as a MAF agent](#use-it-as-a-maf-agent) | Building and driving the agent with the ordinary MAF agent surface (`run` / streaming / threads / workflows) |
| [Bring your own model client](#bring-your-own-model-client) | Injecting an Azure / Foundry / custom-auth chat client |
| [Sessions: one-shot vs continuous](#sessions-one-shot-vs-continuous) | Persistent multi-turn conversations with a shared workspace |
| [Exchanging files with the user](#exchanging-files-with-the-user) | Delivering output files to a host and ingesting uploads |
| [MCP servers](#mcp-servers) | Plugging in stdio / remote MCP servers |
| [Live status updates](#live-status-updates) | Streaming tool progress to a callback or the CLI |

## Guides

### Use it as a MAF agent

`solve()` and `Conversation` are conveniences over a plain Microsoft Agent Framework agent. When you want the agent **itself** — to drive it directly, stream it, give it a multi-turn thread, or drop it into a MAF workflow as a node/sub-agent — build it with `Session.create_agent()` and then use the ordinary MAF agent surface:

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from sheath import Session, SheathConfig

async def main():
    # The two-step (open a workspace, then build the agent over it) is the only sheath-specific
    # part — it binds the agent to one session root + sandbox. The build call itself is standard MAF.
    async with Session.create(SheathConfig()) as session:
        agent = await session.create_agent(
            OpenAIChatClient(model="gpt-6.1-sol", env_file_path=".env"),
            agent_instructions="Reconcile EU sales and flag anomalies.",  # your task prompt
            tools=[query_sales, mcp],          # plain callables + MCPTools, spill-handled for you
            bundles=("code", "files", "web"),  # which built-in tool families to include
        )

        # `agent` is a MAF `Agent` — the usual interface, nothing sheath-specific from here:
        reply = await agent.run("Which regions look anomalous?")
        print(reply.text)

        # Multi-turn over a MAF AgentSession (history threads across calls):
        thread = agent.create_session()
        await agent.run("Now compare against last quarter.", session=thread)

        # Streaming:
        async for update in agent.run("Summarize the findings.", stream=True):
            print(update.text, end="")

asyncio.run(main())
```

Because the returned object is a standard MAF `Agent`, it composes anywhere a MAF agent is expected — as a workflow node, a sub-agent, or behind the AG-UI wrapper (`agui_stream` does exactly this with `conv.agent`). Construction goes through MAF's own `create_harness_agent(client, …)` factory, so apart from the workspace two-step and the spill-wrapping of your tools, you are building and running an ordinary MAF agent.

**Defaults `create_agent` sets.** Sheath turns off the framework's built-in todo, mode, file-memory and web-search providers (the agent's tools come from the bundles you select, not from MAF), and the framework's tool-approval middleware (it would raise unless every `run()` is given an `AgentSession`; sheath has no approval flow, so the returned agent can be `run()` bare). It also sends `store=False` with every request — see [Model-provider storage](#security-and-confinement). You can override `default_options` (your other options are kept; `{"store": True}` opts back in), `disable_tool_auto_approval`, and the token budgets `max_context_window_tokens` / `max_output_tokens` by passing them to `create_agent`; the four provider switches (todo, mode, file memory, web search) are fixed, and passing one raises `TypeError`. The one-shot and conversation paths (`solve`, `aopen`, `agui_stream`) always use the defaults; build the agent with `create_agent` to change them.

### Bring your own model client

The convenience entry points (`solve`, `Sheath(...)` with no `client`) use MAF's OpenAI client by default — following [MAF's own Python convention](https://github.com/microsoft/agent-framework/blob/main/python/README.md) (`OPENAI_API_KEY` present → OpenAI). The sheath is **not** OpenAI-only: it needs only an object implementing MAF's chat-client protocol (`SupportsChatGetResponse`), and never authenticates one for you when you supply it. The same client is reused across one-shots and every continuous conversation. Whatever client you supply, `create_agent` applies the defaults in [Use it as a MAF agent](#use-it-as-a-maf-agent), including `store=False`, a standard MAF chat option. Two injection points:

```python
# High-level facade — your client flows to solve / aopen / agui_stream
h = Sheath(SheathConfig(), client=my_client)

# Composable path — pass it straight to create_agent
agent = await session.create_agent(my_client, agent_instructions="…", tools=[…])
```

MAF's `OpenAIChatClient` already does Azure routing. For Azure with a **refreshing token provider**:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from agent_framework.openai import OpenAIChatClient
from sheath import Sheath, SheathConfig

token_provider = get_bearer_token_provider(      # fresh bearer token per call → handles refresh
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

client = OpenAIChatClient(
    azure_endpoint="https://my-resource.openai.azure.com",
    api_version="2024-10-21",
    credential=token_provider,
    model="my-deployment",                       # Azure deployment name
)
h = Sheath(SheathConfig(), client=client)      # config.model is ignored; the client carries it
```

If you already hold a fully-built MAF chat client (your own custom-auth Azure / Foundry / OpenAI-compatible client), skip all of the above and just pass it: `Sheath(SheathConfig(), client=that)`.

### Sessions: one-shot vs continuous

The sheath serves both one-shot and continuous (multi-turn) interactions from any frontend.

- **One-shot** (`solve`) is ephemeral — it runs and **cleans up its workspace** when done. Pass `keep=True` to retain the audit trail (`result.session_dir`).
- **Continuous** keeps a persistent workspace (handles + sandbox files) and conversation history across turns, until you close it (or an optional idle TTL expires).

```python
import asyncio
from sheath import Sheath, SheathConfig

async def main():
    h = Sheath(SheathConfig(idle_ttl_s=1800))     # idle conversations expire after 30 min (optional)
    conv = await h.aopen("thread-42")                # open or resume by id (e.g. an AG-UI threadId)
    print((await conv.aask("load sales.csv and summarize")).final_text)
    print((await conv.aask("now filter to EU")).final_text)   # sees the prior turn's handles + history
    await conv.aclose()                              # reap this conversation's workspace
    await h.aclose_sessions()                        # (host shutdown) close any remaining
    # await h.sweep_sessions()  # reap idle conversations on your own cadence

asyncio.run(main())
```

**Agent instructions.** Give the agent your domain system prompt with `agent_instructions` — on `Sheath(...)` as the default for every conversation, or per conversation on `aopen(...)` (which wins when both are set); `solve`/`asolve` accept it too. MAF layers it *after* sheath's operating manual: `<core + bundle instructions>`, a blank line, then `<agent_instructions>`. Leave tool mechanics to sheath and put the data model, workflow, and conventions in yours. The per-conversation value applies when a conversation is created; re-opening a live id returns the existing agent.

```python
h = Sheath(cfg, agent_instructions=SYSTEM_PROMPT)
conv = await h.aopen("thread-42", agent_instructions=OTHER_PROMPT)   # optional override
```

Conversation ids (`aopen(id)`, an AG-UI `threadId`) must be a single safe path segment; an empty, `.`/`..`, separator-containing or over-long id is rejected. A conversation with a turn in flight (`aask` or an open AG-UI stream) is never reaped by the idle TTL, and sync `add_input` raises during one (use `aadd_input`).

AG-UI hosts get this automatically — `agui_stream` maps each request's `threadId` to a persistent conversation, so handles and files persist across turns (history stays in the AG-UI message replay).

### Exchanging files with the user

**Delivering files.** Pass `on_publish` (on `Sheath(...)`, or per conversation on `aopen`) and
select the `deliver` bundle. The model writes a file under `outputs/` and calls `publish_file`
(or `publish()` inside `run_python`, or `publish_handle` for a handle). Sheath validates every
request in the parent — inside the root, a regular file and not a symlink, not under
`handles/` or `.scripts/` or a control file, at most `max_publish_bytes` — then calls your
callback with a `PublishedFile(path, rel_path, name, size, sha256, content_type, description,
source)`. `path` is a private snapshot outside the workspace, copied from the validated bytes;
copy it during the callback — it is deleted when the callback returns. Its return
value reaches the model under `"host"`; if it raises, the tool returns an error and the run
continues. The callback runs synchronously in the tool call's context (so your contextvars are
visible) but possibly on a worker thread — make it thread-safe. Sandbox `publish()` requests
are processed after the script exits cleanly. Both the tools and `publish()` require the
`deliver` bundle **and** a callback; without either, they are not available to the model.

```python
from sheath import PublishedFile, Sheath

def host_publish(pf: PublishedFile) -> dict:
    file_id = storage.copy_in(pf.path, name=pf.name, run=CURRENT_RUN.get())
    return {"file_id": file_id, "label": pf.name}

h = Sheath(cfg, bundles=("code", "files", "deliver"), on_publish=host_publish)
```

**Receiving files.** Between turns, hand uploads to the conversation:

```python
handle = conv.add_input(upload_bytes, input_id=upload.id, name=upload.filename)
# or: await conv.aadd_input(...)   # waits for a running turn instead of raising
```

The file lands at `inputs/<input_id>/<name>` (read-only; mounted `:ro` in the container tier)
and becomes a `binary` handle — `load(id)` in `run_python` returns its path. It is idempotent by
`input_id`, survives a restart on the same root, is capped by `max_input_bytes`, and is
**never parsed in the host process** (the preview is name/size/type plus raw first lines for
text files). `conv.inputs` lists them. Note: `read_document` on an input parses it with Docling
**in the host process**; prefer reading uploads inside `run_python` when isolation matters.

### MCP servers

Pass a MAF `MCPTool` — stdio, Streamable-HTTP, or WebSocket — in the same `tools=[…]` list. The sheath connects each server, **owns its lifecycle** (closing it when the run ends, even on error), and attaches the same spill handling to every tool the server exposes — so a large MCP return (a long message list, a big query result) lands as a clean typed handle instead of flooding context.

```python
from agent_framework import MCPStdioTool
from sheath import Sheath, SheathConfig

# A stdio MCP server, launched however your other MCP clients launch it.
# (If the server resolves its config/credentials relative to a directory, launch it
#  from there, e.g. command="sh", args=["-c", "cd /path/to/server && exec uv run its-cmd"].)
mcp = MCPStdioTool(name="msgraph", command="uv", args=["run", "msgraph-mcp"])

h = Sheath(SheathConfig(model="gpt-6.1-sol"))
result = h.solve("List the subjects of my 5 most recent emails.", tools=[mcp])
print(result.final_text)
```

A remote server works the same way:

```python
from agent_framework import MCPStreamableHTTPTool

mcp = MCPStreamableHTTPTool(
    name="my-service",
    url="https://example.com/mcp",
    header_provider=lambda _: {"Authorization": f"Bearer {token}"},  # if it needs auth
)
result = h.solve("…", tools=[mcp])
```

MCP support needs the `mcp` SDK, which is a declared dependency, so `uv sync` already installs it.

**Serving many conversations: `tool_factory`.** A single live `MCPTool` is fine for one run, but it is owned and closed by whichever session finishes first, and every conversation would share its connection. For a host serving many conversations (AG-UI, CopilotKit), wrap the construction so each conversation builds and connects its own server:

```python
from sheath import Sheath, tool_factory
from agent_framework import MCPStdioTool

h = Sheath(tools=[tool_factory(lambda: MCPStdioTool(name="msgraph", command="uv",
                                                    args=["run", "msgraph-mcp"]))])
```

`tool_factory` takes a zero-argument callable; it is called once per conversation. Only MCP-shaped objects are cleaned up on teardown: a session recognizes them by duck-typing on `connect`, `close`, and `.functions`, and closes the ones it connected. If your factory returns some other stateful object (a database client, say), the sheath will not close it — manage its lifecycle yourself.

**On result size.** Spilling is lossless — a large page is preserved whole as a handle, so an MCP server's pagination cursor survives and the agent can fetch the next page. The two ends of the spill-over zone are configurable (`spill_threshold_bytes` … `max_spill_bytes`): a well-behaved server that paginates returns bounded pages that flow through cleanly, while an unbounded dump past `max_spill_bytes` raises `SpillLimitExceeded` rather than silently filling disk — nudging the source toward server-side pagination/filtering.

### Live status updates

Tools report progress while they run, and you receive it through an `on_status` callback — from the built-in tools, from your own tools (via `report_progress`), and from MCP servers (their logging and progress notifications are captured automatically).

```python
from sheath import Sheath, report_progress

def crunch(n: int) -> str:
    """Your tool can report progress."""
    for i in range(n):
        report_progress(f"processed {i + 1}/{n}", current=i + 1, total=n, tool="crunch")
    return "done"

def show(event):
    bar = f" [{event.current}/{event.total}]" if event.current is not None else ""
    print(f"→ {event.tool}: {event.message}{bar}")

Sheath(tools=[crunch], on_status=show).solve("crunch 5 items")
```

The CLI exposes the same feed with `-v`/`--verbose` (printed to stderr). MCP-server status needs no extra wiring — a server's `notifications/message` arrive tagged `mcp:<server>`, and its `notifications/progress` arrive tagged with the calling tool's name plus `current`/`total`.

### AG-UI and CopilotKit

A sheath run can drive an [AG-UI](https://docs.ag-ui.com/) client such as [CopilotKit](https://docs.copilotkit.ai/) — streamed answer text, live tool-call visibility, and the status feed above — all as AG-UI events. Install the optional extra:

```bash
uv sync --extra agui
```

`Sheath.agui_stream(input_data)` yields AG-UI events for one request; encode them as SSE from your endpoint:

```python
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sheath import Sheath

app, sheath = FastAPI(), Sheath()

@app.post("/agent")
async def agent(request: Request):
    input_data = await request.json()                 # AG-UI RunAgentInput
    encoder = EventEncoder()
    async def sse():
        async for event in sheath.agui_stream(input_data):
            yield encoder.encode(event)
    return StreamingResponse(sse(), media_type="text/event-stream")
```

Point your AG-UI client at this endpoint. The sheath reuses the official `agent-framework-ag-ui` converter, so **frontend/generative-UI tools** (defined in the request), **shared state** (`state_schema`/`predict_state_config`, forwarded as keyword args to `agui_stream`), **human-in-the-loop**, and **multi-turn history** all work — with the sheath's own progress feed overlaid as `sheath.status` `CUSTOM` events. A runnable version is in [`examples/agui_server.py`](https://github.com/jspv/sheath/blob/main/examples/agui_server.py).

Each request is one conversation turn: turns on the same `threadId` are serialized, and a turn counts as activity for `idle_ttl_s`. Close the event stream on client disconnect (Starlette's `StreamingResponse` does) so the conversation is released for the next turn; a stream a server never closes keeps the conversation busy, and therefore exempt from the idle TTL. While a stream is open, sync `Conversation.add_input` raises — use `aadd_input`, which waits for the turn. The `threadId` selects the workspace, so authenticate and authorize it in your endpoint before calling `agui_stream`.

## Reference

### Tool surface

The agent gets a small, fixed set of root-confined tools (listed below; the web and document tools activate when their key/extra is present). Domain data sources are *your* tools/MCP servers, auto-handled by spill.

| Tool | Purpose |
|---|---|
| `write_file(path, content)` | Write a file (root-jailed) |
| `read_file(path, offset=0, limit=2000)` | Bounded, paginated text |
| `list_files(path=".")` | Directory listing |
| `search(pattern, …)` | ripgrep over files/handles, with a Python-regex fallback (run in a bounded child process) |
| `run_python(code=None, path=None, args=[])` | Run a script in the sandbox → `{stdout, result, error, exit_code, new_handles}`. An emitted result whose JSON exceeds `spill_threshold_bytes` comes back as `{"spilled": true, "note": …, "handle": {…}}` and is stored as a handle |
| `fetch_url(url)` | Fetch a URL → typed handle; HTML is cleaned to Markdown (trafilatura) |
| `inspect_handle(id, rows=20, stats=False)` | Deeper on-demand look at a handle: up to 100 preview lines / head rows, read raw and bounded (JSON is not pretty-printed); `stats=True` adds per-column min/max/null_count for dataframes from the parquet footer (min/max are null when any non-empty row group lacks them); a first row group over 16 MiB yields no head rows (`head_omitted`) |
| `web_search(query, max_results=5)` | Tavily web search (needs `TAVILY_API_KEY`) |
| `web_extract(url)` | Tavily clean-content extraction (needs `TAVILY_API_KEY`) |
| `read_document(source)` | A workspace path or URL → clean markdown handle (tables preserved) via Docling; needs the `docling` extra. A URL download over `FetchConfig.max_bytes` is refused, not truncated; a workspace path is copied privately and capped at `SheathConfig.max_input_bytes` |
| `publish_file(path, name=None, description=None)` | Deliver a workspace file to the host (`deliver` bundle; only with `on_publish`) |
| `publish_handle(handle_id, format=None, name=None)` | Deliver a handle as csv/xlsx/parquet/json (`deliver` bundle; only with `on_publish`) |

### Security and confinement

Every session has one **root directory**; everything — handles, agent-written scripts, reads/writes, the sandbox `cwd` — lives under it.

- **Layer 1 — Tool path-jail (guaranteed).** All model-supplied paths route through one chokepoint, `safe_path(root, p)`, which resolves symlinks *before* checking and rejects any path outside the root (blocks `..`, absolute paths, symlink escapes). It's the most heavily tested code in the project.
- **Layer 2 — Executed code.** `run_python` runs in the **container** tier by default: a hardened Podman/Docker container (network off, read-only root filesystem, dropped capabilities, non-root, memory/cpu/pid limits) — see [Sandbox tiers](#sandbox-tiers). If no usable container runtime is present, building the sandbox is a hard error (`SandboxRuntimeUnavailable`) that names the `local` opt-out; it never falls back silently. **The `local` tier is not a security boundary**: the code runs as the host user, with that user's file and network access, in a scrubbed-env subprocess with `resource` rlimits and a wall-clock timeout. Use it only when you have chosen that deliberately. A host that must not be able to opt out sets `SandboxConfig.require_isolation=True`, which refuses the local tier however it was selected — see [Sandbox tiers](#sandbox-tiers).
- **Workspace trust zones.** The session root is bind-mounted into the sandbox, so the harness never trusts what it finds directly under it. Four directories are owned by the harness and mounted **read-only** into the container: `handles/` (handle files), `inputs/` (uploads), `.scripts/` (inline scripts) and `.sheath/` (the manifest, scratch files and per-run exchange directories). Sandboxed code can neither write inside them nor replace them, and `write_file` refuses paths that resolve into them. Each run gets one writable exchange directory (`/sheath-run`): `save()`, `emit()` and `publish()` write there, and the harness reads it back by fixed file name with no-follow, regular-file-only, size-capped opens, then **moves** saved payloads into `handles/` itself. Nothing sandboxed code wrote is ever followed: a `publish()` request's path is joined to the root lexically and then walked component by component with `lstat`, refusing any link, and the harness's own `outputs/`, `.sheath/tmp/` and `.sheath/runs/` are required to be real directories (a link planted at `outputs/` is replaced by a real directory, never written through). Before and after every run the harness checks that the four directories, `.sheath/tmp/` and `.sheath/runs/` are still real directories and raises `WorkspaceTamperedError` otherwise.
- **Orphan quarantine.** If a container-tier run times out and the runtime cannot confirm the container was removed, it may still be writing in the root. The session is then quarantined: every later `run_python` call and every file tool (`read_file`, `write_file`, `list_files`, `search`, `publish_file`, `publish_handle`, `read_document` on a workspace path, `add_input`) first retries `<runtime> rm -f <that exact container>` and, until the runtime confirms it is gone, refuses with `SandboxQuarantinedError` (the publish and document tools return it as an `error`) without touching the root. Only the recorded container names are ever removed.
- **Bounded search fallback.** Without ripgrep, `search` scans in a separate Python process that is killed after 30 s (like the ripgrep path), so a catastrophic model-supplied regex cannot freeze the host process.
- **Layer 3 — Egress.** The model chooses the URLs the harness fetches, so egress is filtered. `fetch_url` and `read_document(url)` deny loopback, private, link-local, reserved, multicast and unspecified addresses by default, plus `100.64.0.0/10` (RFC 6598 carrier-grade NAT), which no standard `ipaddress` flag reports as private and so is denied explicitly. The harness follows redirects itself (`follow_redirects=False` is forced on every request), so **every hop is re-validated**, including its scheme against `FetchConfig.allowed_schemes`. Bodies are streamed: reading stops at the byte limit, a model-supplied `max_bytes` is clamped to `FetchConfig.max_bytes_ceiling`, and the whole fetch is bounded by `FetchConfig.total_timeout_s`. `read_document` downloads through the same guard and hands Docling a local path, never a URL, so Docling cannot follow its own redirects outside policy. To reach an internal data source, name it in `FetchConfig.allow_private_hosts` (hostnames or CIDRs).
  - **Cloud metadata endpoints are denied unconditionally and cannot be allowlisted.** `allow_private_hosts` cannot open them, by design: `169.254.169.254`, `169.254.170.2`, `168.63.129.16`, `100.100.100.200`, `192.0.0.192` and `fd00:ec2::254`. Note `168.63.129.16` (Azure's wireserver) is a *public* address that nothing else would block; Azure users should expect it to be refused.
  - Known residual: the total deadline is enforced by a watchdog that shuts down the connections opened during the call; on expiry `guarded_fetch` raises `FetchDeadlineExceeded` (an `httpx.HTTPError`). It does **not** bound DNS resolution inside `validate_url`, and the TCP/TLS connect is bounded only by the client's per-phase timeout (`FetchConfig.timeout_s`).
  - Known residual: the host is resolved for validation and then resolved again by the HTTP client, so a DNS-rebinding race is not closed (IP pinning is on the [roadmap](https://github.com/jspv/sheath/blob/main/docs/ROADMAP.md)). `web_search` / `web_extract` call a fixed Tavily endpoint and are not filtered; an MCP server you install makes its own network calls, which the harness cannot filter.
- **Handle integrity.** Metadata the model sees for a handle (schema, row count, preview) is derived by the harness from the bytes on disk **when the handle is created**, never reported by the sandboxed code that wrote them — and it is re-derived, not believed, when a manifest is rehydrated onto a root that outlived its process. The handle **record** is immutable: reusing an existing id is refused and reported. In the container tier the handle **bytes** are immutable too, because `handles/` is a read-only mount. As defence in depth (and as the only tamper signal in the `local` tier), a content digest is recorded when the metadata is derived and checked on every read; a mismatch raises `HandleTamperedError`. The digest (sha256, version-tagged) covers the file's length, its first 64 KiB and its last 64 KiB, so it is not a whole-file check. Handle ids chosen by sandboxed code must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. The parent↔sandbox exchange is bounded (1 MiB emit payload, 8 MiB control file, 256 new handles per run; see the configuration table).
- **Bounded results and reads.** Every built-in tool result has a ceiling: `run_python` keeps the head and tail of `stdout`/`stderr` (`SandboxConfig.max_output_bytes` each) and stores a large emitted result as a handle; `inspect_handle`, `read_file`, `list_files` and `search` cap what they return. The harness reads sandbox-written and fetched data with bounded memory: files are streamed, only regular files are opened (a FIFO cannot block it), a dataframe handle is never materialized in the host process (previews read one record batch, `inspect_handle(stats=True)` reads footer statistics, and `publish_handle` converts inside the sandbox), and HTTP bodies are streamed up to the byte limit under a total deadline. Document inputs are bounded too: a `read_document` URL download is capped at `FetchConfig.max_bytes` and refused, not truncated, when larger, and a workspace path is copied privately and capped at `SheathConfig.max_input_bytes`. A dataframe whose first row group exceeds 16 MiB gets no head rows or preview (`head_omitted` / "preview omitted"), because Arrow would decode the whole group.
- **Residual risks.** The harness still parses the parquet footer and first record batch of a file sandboxed code wrote, and an oversized data page is decoded whole. Container-tier disk use is unbounded (`max_file_size_mb` applies to the `local` tier only). DNS rebinding between validation and connect is not closed, and the fetch deadline does not cover DNS resolution (see Layer 3). The `local` tier provides none of the guarantees in this section.
- **Model-provider storage.** The built-in OpenAI client stores conversations at OpenAI by default. Because this agent sends fetched web pages, documents and workspace data to the model, `create_agent` sends `store=False`, so conversation history is kept in-process and OpenAI is asked not to retain it for later retrieval. That is a request to the provider, not a guarantee of zero retention: the provider's own logging and your account's data controls still apply, and the content is still sent to the model. Opt back in per agent with `default_options={"store": True}`.
- **Tool isolation.** Wrap stateful tools such as MCP servers in `tool_factory` so each conversation gets its own instance (see [MCP servers](#mcp-servers)).

> The **container tier** is the default and provides real isolation. Select `local` explicitly with `SheathConfig.sandbox.backend = "local"` (or `SHEATH_SANDBOX_BACKEND=local`) to opt out; both sit behind the same `SandboxExecutor` interface.

### Sandbox tiers

`run_python` executes model-authored code; two backends are chosen by `SheathConfig.sandbox.backend`. **`container` is the default**; `local` is an explicit opt-out.

- **`container`** (default) — runs the code in a hardened OCI container (Podman preferred, Docker supported; auto-detected). Network **off by default**, read-only root filesystem, `--cap-drop ALL`, non-root, and memory/cpu/pid limits. The session root is bind-mounted to `/workspace` with `handles/`, `inputs/`, `.scripts/` and `.sheath/` overlaid read-only, plus one writable per-run exchange directory at `/sheath-run`; a script that writes into `handles/` directly gets a read-only error — use `save(id, obj)`. If no usable runtime is found (a binary on `PATH` is not enough — the check probes liveness with `<runtime> info`), building the sandbox raises `SandboxRuntimeUnavailable`, with a message naming the `local` opt-out.
- **`local`** (explicit opt-out) — a scrubbed-env subprocess with `resource` rlimits, a wall-clock timeout, and the path-jail. When a run ends (normally or by timeout) the harness kills the child's whole process group, so background processes a script starts do not outlive the run. Fast, no dependencies, but **no isolation**: the code runs as the host user and can read or write anything that user can, and use the network. Selecting it emits `NoSandboxIsolationWarning`; filter it with `warnings.filterwarnings("ignore", category=sheath.NoSandboxIsolationWarning)` once you have decided deliberately (CI, tests).

```python
from sheath import Sheath, SheathConfig
from sheath.config import SandboxConfig

cfg = SheathConfig(sandbox=SandboxConfig(
    backend="container",           # the default; "local" opts out of isolation
    network=False,                 # opt-in with True if sandbox code must reach the network
    pip_packages=("rich",),        # provisioned into a mounted layer (network only during provisioning)
))
Sheath(cfg).solve("…")
```

**Requiring isolation.** The container tier is already the default, so `require_isolation=True` is the stricter assertion on top of it: it refuses the local tier *however it was selected* — an explicit `backend="local"`, or `SHEATH_SANDBOX_BACKEND=local` in the environment — with `SandboxRuntimeUnavailable` at `Session.create`, not on first use. A missing runtime or image raises; sheath never falls back to the local tier. Set it on any host where no deployment mistake may quietly turn isolation off.

**Image readiness.** Runtime liveness and image readiness are different questions. Selecting the container tier probes the runtime with `<runtime> info` and raises `SandboxRuntimeUnavailable` if it is absent or not responding; whether the *image* is built is checked separately. By default the image (Python + `preinstalled` libraries) and any `pip_packages` layer are **built on first use** and cached — which, with `container` as the default backend, is what an out-of-the-box run does. `build_on_demand` controls this; its default (`None`) means "build unless `require_isolation`", so a host that demands isolation is also required to pre-build rather than discover a build on the first request. With builds disabled, a missing image makes `run_python` fail immediately with `SandboxImageMissing`, naming the exact command to run. Builds that do run are bounded by `build_timeout_s` (`SandboxImageError` on expiry); it is one
budget for the image build and the pip layer together. A timed-out pip-layer container is
removed, but a timed-out image build may finish in the runtime's background. Pre-build exactly what a host will ask for, and gate startup on a non-mutating preflight:

```bash
sheath-build-sandbox --preinstalled pandas pyarrow numpy httpx --pip rich   # build image + layer
sheath-build-sandbox --pip rich --check                                    # report only; exit 1 on problems
```

```python
from sheath import sandbox_preflight
from sheath.config import SandboxConfig

cfg = SandboxConfig(backend="container", require_isolation=True)   # build_on_demand -> False
report = sandbox_preflight(cfg)        # never builds; `image inspect` is time-bounded
if not report.ok:
    raise RuntimeError("; ".join(report.problems))
```

Notes: on macOS the runtime runs in a Linux VM, so the session root must sit under a VM-shared path (the default root is under the current directory, so run from somewhere inside your home directory, or set `root_dir`); the container tier does not enforce `max_file_size_mb` (memory/pid/cpu/network are enforced instead). The explicit `network=True` option exposes the container to the network; the egress guard applies to the harness's own fetches, not to code running in the sandbox.

### Configuration

`SheathConfig` (see `sheath/config.py`) is a plain dataclass:

| Field | Default | Notes |
|---|---|---|
| `model` | `"gpt-6.1-sol"` | Used only by the built-in OpenAI client; ignored when you inject a `client` |
| `spill_threshold_bytes` | `8192` | Lower edge of the spill-over zone: a tool return over this becomes a handle |
| `max_spill_bytes` | `100 MiB` | Upper edge: a return larger than this is **rejected loudly** (`SpillLimitExceeded`), never silently stored |
| `max_context_window_tokens` | `128_000` | Fed to MAF compaction |
| `max_output_tokens` | `4096` | |
| `root_dir` | `None` | `None` → a fresh one-shot dir `./.sheath/runs/run-XXXXXXXX`; continuous conversations use `./.sheath/sessions/<id>` (or `<root_dir>/<id>`) |
| `idle_ttl_s` | `None` | Continuous-session idle TTL; `None` → never expire |
| `max_publish_bytes` | `100 MiB` | Largest file `publish_file` / `publish()` will deliver |
| `max_input_bytes` | `100 MiB` | Largest upload `add_input` will accept |
| `sandbox` | `SandboxConfig()` | `backend` (`container` by default; `local` = no isolation), `container_runtime`, timeout, limits, network, `pip_packages`, preinstalled libs, `require_isolation` (`False` — refuses a non-container backend however it was selected), `build_on_demand` (`None` → build unless isolation is required), `build_timeout_s` (`900`). **`max_file_size_mb` is enforced by the `local` tier only** — under the default `container` backend it has no effect (memory/cpu/pid limits and the network setting apply instead) |
| `fetch` | `FetchConfig()` | `max_bytes`, timeout, allowed URL schemes |
| `sandbox.max_output_bytes` | `8192` | Bytes of `stdout` and of `stderr` kept from a `run_python` call (first and last halves; the gap is reported) |
| `fetch.total_timeout_s` | `120` | Wall-clock bound on one fetch, all redirect hops included |
| `fetch.max_bytes_ceiling` | `100 MB` | Upper bound on the per-call `max_bytes` the model may pass to `fetch_url` |
| `fetch.allow_private_hosts` | `()` | Hostnames or CIDRs exempted from the internal-address denylist, for internal data sources. Cloud metadata endpoints cannot be allowlisted |
| `fetch.max_redirects` | `5` | Redirect hops followed (each re-validated); more raises an error |
| `max_emit_bytes` | `1 MiB` | Cap on a sandbox emit payload, checked before parsing |
| `max_control_bytes` | `8 MiB` | Read cap on the sandbox's new-handles control file |
| `max_new_handles` | `256` | Handle records adopted per `run_python` call |
| `search` | `SearchConfig()` | Tavily provider, key, `max_results` |
| `documents` | `DocumentConfig()` | Docling ingestion: `ocr` (off by default) |

**`SHEATH_SANDBOX_BACKEND`.** An environment variable that overrides the default of `SandboxConfig.backend`. It selects the **tier** (`container` or `local`), *not* the container runtime — use `SandboxConfig.container_runtime` to choose podman or docker. Setting it to `local` opts out of isolation; that is intended for CI and test environments that have made the choice deliberately. An explicit `SandboxConfig(backend=...)` always wins over the variable. Any value other than exactly `local` or `container` is an error (`ValueError`), not a silent fallback.

## Development

```bash
uv sync --extra agui --extra docling   # the extras the full suite exercises
uv run pytest                          # full suite — no network or API keys required
uv run ruff check .
```

With every extra installed, a few tests still skip: the three `search` tests that exercise ripgrep skip when `rg` is not on `PATH`, and the container-tier tests skip when no container runtime is usable. Without the `agui` extra the AG-UI tests skip too.

CI (`.github/workflows/tests.yml`) runs lint and the suite on every push and pull request, and publishing requires it. The container-tier tests run there when a runtime is usable (ubuntu-latest has one) and build the sandbox image; they skip only when none is.

Testing is a first-class principle: **tests are written before implementation for every unit** (TDD). The fast suite runs deterministically against a stub chat client (no API cost); live tests are opt-in and gated by environment variables. `safe_path` and the spill logic are held to the highest bar.

**Live tests** run against real services and are skipped unless you opt in:

| Variable | Needs | Cost |
|---|---|---|
| `SHEATH_LIVE=1` | `OPENAI_API_KEY` (and `TAVILY_API_KEY` for the web test) in `.env` | paid model calls |
| `SHEATH_LIVE_AGUI=1` | the `agui` extra and `OPENAI_API_KEY` | paid model calls |
| `SHEATH_LIVE_DOCLING=1` | the `docling` extra (models download on first use) | local, no API cost |

```bash
SHEATH_LIVE=1 SHEATH_LIVE_AGUI=1 SHEATH_LIVE_DOCLING=1 uv run pytest
```

Live tests that run code (`run_python`) use the default container sandbox, so they also need a usable container runtime.

**Evals** run the agent on 10 find-and-analyze tasks (some fetch live web pages, so answers vary) and write each task's final answer, any error, the tools used and the status events as JSON. They do not grade the answers, so read them. `SHEATH_EVAL_MODEL` picks the model (default: the shipped default) and `SHEATH_EVAL_OUT` picks where `results/` and `runs/` go, so runs for different models do not overwrite each other:

```bash
SHEATH_EVAL_MODEL=gpt-5-mini SHEATH_EVAL_OUT=/tmp/evals/gpt-5-mini uv run python evals/run_evals.py [task_id ...]
```

Design specs and phased implementation plans for each feature live under [`docs/superpowers/`](https://github.com/jspv/sheath/tree/main/docs/superpowers).

## Project layout

```
sheath/
  config.py      SheathConfig / SandboxConfig / FetchConfig / SearchConfig / DocumentConfig
  paths.py       safe_path() — the single security chokepoint
  workspace.py   trust zones: protected dirs, per-run exchange dir, bounded no-follow reads
  procio.py      child-process runner with bounded output capture
  handles.py     Handle + HandleStore (json / text / dataframe persistence)
  sandbox.py     SandboxExecutor protocol + LocalSubprocessSandbox
  sandbox_container.py  hardened OCI-container backend (podman/docker)
  container_runtime.py  runtime detection, image build, package layer
  session.py     Session — root dir + store + sandbox for one run
  conversation.py  persistent multi-turn Conversation (workspace + MAF AgentSession)
  manager.py       SessionManager: continuous-session registry + lazy TTL
  spill.py       large/structured tool & MCP returns → handles (MAF result_parser)
  status.py      StatusEvent / StatusBus: the live status feed
  mcp_status.py  MCP server logging + progress notifications → status events
  publish.py     publishing workspace files to the host: types and parent-side validation
  agui.py        AG-UI event stream (status overlay over agent-framework-ag-ui)
  bundles.py     capability bundles (code / files / web) + their instructions
  egress.py      egress policy: internal-address denylist + redirect re-validation
  testing.py     StubChatClient: a deterministic chat client for tests (no API)
  tools/         the agent's tools (files, search, fetch, code, inspect, web, documents, deliver);
                 registry.py wraps the Session-bound implementations as agent-ready callables
  runtime/       in-sandbox helpers (load/save/emit/publish), the script runner and the Containerfile
  api.py         Sheath / solve() / Result
  cli.py         thin streaming CLI
docs/superpowers/  design specs + phased implementation plans
examples/          runnable samples (minimal AG-UI server; full CopilotKit + MCP app)
evals/             10-task eval harness for comparing models (see Development)
tests/             mirror of the package (unit + integration + security tests)
```

## Roadmap

**Implemented:** foundation (handles, sandbox, path-jail); the agent loop + tool surface; the `Sheath`/`solve()` API + CLI; web research (Tavily search/extract, Markdown fetch); **document ingestion** (`read_document` via Docling — PDF/spreadsheet → Markdown with tables); **live status updates** (built-in, developer, and MCP tools → an `on_status` feed / `--verbose`); an **AG-UI / CopilotKit** integration (`Sheath.agui_stream`); a **container sandbox tier** (hardened Podman/Docker isolation behind `SandboxExecutor`); and a **session lifecycle** model (ephemeral one-shot `solve`; persistent continuous `aopen`/`aask`/`aclose` with optional idle TTL).

The **security-hardening phase** (zero-trust handle metadata, container-by-default, egress guard, per-conversation tools) and the **workspace trust zones** work (read-only harness state in the sandbox, per-run exchange directory, bounded tool results and reads) are delivered; see [`docs/ROADMAP.md`](https://github.com/jspv/sheath/blob/main/docs/ROADMAP.md) for residuals.

**Planned** (documented under [`docs/superpowers/`](https://github.com/jspv/sheath/tree/main/docs/superpowers)):

- **Micro-VM sandbox tier** — gVisor / Firecracker, behind the same `SandboxExecutor` interface.
- **MAF skills + memory providers** (v1.1) and a MAF Workflow durability / HITL outer shell.

**Out of scope for now:** headless-browser/JS rendering, a full search-provider abstraction, and alternative document backends.
