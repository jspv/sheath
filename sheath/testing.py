"""StubChatClient: a deterministic chat client for testing the agent loop without an API.

Drive it with a script of `tool_call(...)` / `text(...)` steps; each model turn pops the
next step. A tool_call step makes the agent invoke that tool; a text step ends the run. A step
may also be a callable taking that turn's messages and returning the Content to answer with,
e.g. to echo the tool results the model was given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
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


class StubChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Returns scripted responses, one per model turn.

    Layered like the framework's real clients (function invocation over chat middleware), so
    chat middleware an agent wires (message injection among it) runs around each model call."""

    def __init__(self, script: list[Union[Content, Callable[[list[Message]], Content]]]) -> None:
        super().__init__()
        self._script = list(script)
        self._turn = 0

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # Plain (non-async) method per the BaseChatClient contract: return an awaitable ChatResponse
        # when stream=False, or a ResponseStream when stream=True (the AG-UI path uses streaming).
        step = self._script[min(self._turn, len(self._script) - 1)]
        self._turn += 1
        if callable(step):
            step = step(list(messages))
        if stream:
            async def _updates():
                yield ChatResponseUpdate(role="assistant", contents=[step])

            return self._build_response_stream(_updates())

        async def _response() -> ChatResponse:
            return ChatResponse(messages=Message(role="assistant", contents=[step]))

        return _response()
