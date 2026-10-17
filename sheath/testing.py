"""StubChatClient: a deterministic chat client for testing the agent loop without an API.

Drive it with a script of `tool_call(...)` / `text(...)` steps; each model turn pops the
next step. A tool_call step makes the agent invoke that tool; a text step ends the run.
"""

from __future__ import annotations

from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    FunctionInvocationLayer,
    Message,
)
from agent_framework._types import Content


def tool_call(name: str, arguments: dict[str, Any] | None = None, call_id: str = "c") -> Content:
    return Content("function_call", call_id=call_id, name=name, arguments=arguments or {})


def text(value: str) -> Content:
    return Content("text", text=value)


class StubChatClient(FunctionInvocationLayer, BaseChatClient):
    """Returns scripted responses, one per model turn."""

    def __init__(self, script: list[Content]) -> None:
        super().__init__()
        self._script = list(script)
        self._turn = 0

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # Plain (non-async) method per the BaseChatClient contract: return an awaitable ChatResponse
        # when stream=False, or a ResponseStream when stream=True (the AG-UI path uses streaming).
        step = self._script[min(self._turn, len(self._script) - 1)]
        self._turn += 1
        if stream:
            async def _updates():
                yield ChatResponseUpdate(role="assistant", contents=[step])

            return self._build_response_stream(_updates())

        async def _response() -> ChatResponse:
            return ChatResponse(messages=Message(role="assistant", contents=[step]))

        return _response()
