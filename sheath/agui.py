"""AG-UI adapter: run the sheath as an AG-UI event stream, overlaying StatusBus events.

The official ``agent-framework-ag-ui`` package already maps a streaming MAF agent to AG-UI
events (text, tool calls, frontend tools, state, HITL). We reuse it and *overlay* the sheath's
own ``StatusBus`` -- mid-tool progress and MCP logging/progress that the official converter does
not carry -- as ``CustomEvent``s. ``agent-framework-ag-ui`` is an optional extra; this module
imports it (and ``ag_ui``) lazily so the sheath core never depends on it.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing, nullcontext
from typing import Any, AsyncIterator, Callable

from .status import StatusBus, StatusEvent


def status_to_agui(event: StatusEvent) -> Any:
    """Map a StatusEvent to an AG-UI ``CustomEvent(name="sheath.status", value=...)``.

    Lazily imports ``ag_ui`` so importing this module does not require the agui extra.
    """
    from ag_ui.core import CustomEvent

    return CustomEvent(
        name="sheath.status",
        value={"tool": event.tool, "message": event.message,
               "current": event.current, "total": event.total},
    )


async def merge_status(events: AsyncIterator[Any], bus: StatusBus,
                       to_event: Callable[[StatusEvent], Any] = status_to_agui) -> AsyncIterator[Any]:
    """Yield ``events``, interleaving ``bus``'s StatusEvents (mapped via ``to_event``).

    StatusEvents may be emitted from a worker thread (sync tools run via ``asyncio.to_thread``),
    so the subscriber marshals onto the loop with ``call_soon_threadsafe`` (an ``asyncio.Queue``
    is not thread-safe). Queued status events are flushed at each source-event boundary and once
    the source ends. The subscriber is always removed on exit.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def _sink(status_event: StatusEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, to_event(status_event))

    unsubscribe = bus.subscribe(_sink)
    try:
        # Close the source deterministically on early exit (it may be a plain async iterable).
        with_close = aclosing(events) if hasattr(events, "aclose") else nullcontext(events)
        async with with_close as source:
            async for event in source:
                while not queue.empty():
                    yield queue.get_nowait()
                yield event
        await asyncio.sleep(0)          # let any last call_soon_threadsafe put land
        while not queue.empty():
            yield queue.get_nowait()
    finally:
        unsubscribe()


def _call_ids(m: dict[str, Any]) -> list[str]:
    return [tc.get("id") for tc in (m.get("toolCalls") or m.get("tool_calls") or []) if tc.get("id")]


def _result_id(m: dict[str, Any]) -> str | None:
    return m.get("toolCallId") or m.get("tool_call_id")


def repair_tool_message_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder replayed AG-UI messages into a provider-valid tool-call/result sequence.

    Some AG-UI clients (e.g. CopilotKit) replay each backend tool *result* message *before* the
    assistant message that declared the calls. The OpenAI Responses/Chat APIs require the
    assistant ``tool_calls`` message first, immediately followed by its ``tool`` results -- the
    inverted order makes a ``function_call`` look like it has "no tool output" and the request
    400s. This rebuilds a valid ordering: each assistant tool-call message is emitted, then its
    results (in call order). Tool calls with no matching result are dropped (and an assistant
    message that becomes empty is dropped); orphan tool-result messages are dropped. Non-tool
    messages keep their relative order. Handles both camelCase (AG-UI) and snake_case keys.
    """
    results_by_id: dict[str, list[dict[str, Any]]] = {}
    for m in messages:
        if m.get("role") == "tool" and _result_id(m):
            results_by_id.setdefault(_result_id(m), []).append(m)  # type: ignore[arg-type]

    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue                                       # re-emitted right after its assistant call
        if role == "assistant" and _call_ids(m):
            keep = [tc for tc in (m.get("toolCalls") or m.get("tool_calls"))
                    if tc.get("id") in results_by_id]
            if not keep:
                if not m.get("content"):
                    continue                               # assistant message was only orphan calls
                out.append({k: v for k, v in m.items() if k not in ("toolCalls", "tool_calls")})
                continue
            m = {**m, ("toolCalls" if "toolCalls" in m else "tool_calls"): keep}
            out.append(m)
            for tc in keep:
                out.extend(results_by_id[tc["id"]])        # results immediately after the call
            continue
        out.append(m)
    return out


async def agui_event_stream(agent: Any, bus: StatusBus, input_data: dict[str, Any],
                            **agui_kwargs: Any) -> AsyncIterator[Any]:
    """Run ``agent`` via ``AgentFrameworkAgent(**agui_kwargs)`` and overlay ``bus``'s status.

    ``**agui_kwargs`` (e.g. ``state_schema``, ``predict_state_config``, ``require_confirmation``,
    ``name``, ``description``) pass straight to ``AgentFrameworkAgent``, and ``input_data``
    (messages, ``tools``, ``state``) passes to its ``.run`` -- so frontend tools, shared state,
    HITL, and multi-turn all work. The replayed messages are first run through
    ``repair_tool_message_order`` so a client that emits tool results before their tool-call
    message does not 400 the provider. Raises a clear error if the agui extra is missing.
    """
    try:
        from agent_framework_ag_ui import AgentFrameworkAgent
    except ImportError as e:
        raise RuntimeError(
            "AG-UI support unavailable: install the 'agui' extra (e.g. `uv sync --extra agui`)"
        ) from e

    messages = input_data.get("messages")
    if messages:
        input_data = {**input_data, "messages": repair_tool_message_order(messages)}
    afa = AgentFrameworkAgent(agent, **agui_kwargs)
    async with aclosing(merge_status(afa.run(input_data), bus, status_to_agui)) as merged:
        async for event in merged:
            yield event
