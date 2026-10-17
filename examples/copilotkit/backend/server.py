"""CopilotKit / AG-UI backend for the sheath, with an MCP server wired in.

The browser talks to a CopilotKit Next.js app, which proxies each request to this
endpoint as an AG-UI ``RunAgentInput``. ``sheath.agui_stream`` resolves the
request's ``threadId`` to a persistent conversation and streams AG-UI events back.

The demo ``sales`` MCP server (``mcp_server.py``) is attached per request — see
``_sales_mcp`` for why a fresh instance per request is the right pattern.

Run (with the agui extra installed and OPENAI_API_KEY in your environment/.env):
    uv sync --prerelease=allow --extra agui
    uv run uvicorn examples.copilotkit.backend.server:app --reload --port 8000
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ag_ui.encoder import EventEncoder
from agent_framework import MCPStdioTool
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from sheath import Sheath, SheathConfig

app = FastAPI()

# Put session/sandbox output in a temp dir, NOT the default ./.sheath under the repo.
# Why: the agent's run_python writes scripts into the session dir; with `uvicorn --reload`
# watching the repo, those writes would hot-reload the server mid-conversation — aborting the
# in-flight response and wiping the in-memory SessionManager. Keeping the session root outside
# the watched tree avoids that; --reload then only restarts on actual source edits.
_SESSIONS = Path(tempfile.gettempdir()) / "sheath-copilotkit-sessions"
sheath = Sheath(SheathConfig(root_dir=_SESSIONS))  # default OpenAI client (needs OPENAI_API_KEY)

_MCP_SERVER = Path(__file__).with_name("mcp_server.py")


def _sales_mcp() -> MCPStdioTool:
    """A fresh MCP client per request.

    ``agui_stream`` opens one persistent ``Conversation`` per ``threadId``, and that
    conversation connects the MCP server and owns its lifecycle (closing it on
    teardown). A new thread's first message connects this instance; reused threads
    return the existing conversation and ignore the (still-unconnected) fresh object,
    which is simply garbage-collected. Sharing one connected instance across threads
    would let the first thread to finish close the server out from under the others.
    """
    return MCPStdioTool(name="sales", command=sys.executable, args=[str(_MCP_SERVER)])


@app.post("/agent")
async def agent(request: Request) -> StreamingResponse:
    input_data = await request.json()      # AG-UI RunAgentInput (messages, threadId, runId, tools, state)
    encoder = EventEncoder()

    async def sse():
        # agui_stream repairs the replayed tool-call/result ordering for us (CopilotKit emits
        # tool results before their call, which the OpenAI API would otherwise reject).
        async for event in sheath.agui_stream(input_data, tools=[_sales_mcp()]):
            yield encoder.encode(event)    # -> "data: {json}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
