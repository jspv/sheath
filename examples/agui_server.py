"""Serve the sheath to an AG-UI client (e.g. CopilotKit) over SSE.

Run with the agui extra installed:
    uv sync --prerelease=allow --extra agui
    uv run uvicorn examples.agui_server:app --reload
Point your AG-UI client's runtime URL at  http://localhost:8000/agent
"""

from __future__ import annotations

from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from sheath import Sheath

app = FastAPI()
sheath = Sheath()


@app.post("/agent")
async def agent(request: Request) -> StreamingResponse:
    input_data = await request.json()          # AG-UI RunAgentInput (messages, threadId, runId, tools, state)
    encoder = EventEncoder()

    async def sse():
        async for event in sheath.agui_stream(input_data):
            yield encoder.encode(event)         # -> "data: {json}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
