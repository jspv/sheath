"""Translate MCP server notifications (logging + progress) into sheath StatusEvents.

MAF surfaces neither MCP logging nor MCP progress to the sheath, so we hook the (private)
seams its ``MCPTool`` passes to the underlying mcp ``ClientSession``: ``logging_callback`` and
``message_handler``. We wrap them, chaining to the originals to preserve MAF's own behavior
(notably the ``notifications/tools/list_changed`` reload handled inside ``message_handler``).

Servers only emit progress when a ``progressToken`` is on the request, and MAF sets none, so
``inject_progress_tokens`` writes a stable per-tool token into MAF's ``_tool_call_meta_by_name``;
the matching token->tool-name map lets the progress wrapper attribute each event to the right
tool. Every seam is feature-detected -- a future MAF that moves a seam degrades to "no MCP
status", never an error. Emitting is best-effort: a translation error never breaks the wrapped
handler or the tool call.
"""

from __future__ import annotations

from typing import Any, Callable

from .status import StatusBus, StatusEvent


def _is_progress(root: Any) -> bool:
    """Duck-typed check for an mcp ProgressNotification (avoids importing mcp types here)."""
    return getattr(root, "method", None) == "notifications/progress" and hasattr(root, "params")


def _wrap_logging(original: Callable, bus: StatusBus, server: str) -> Callable:
    async def logging_callback(params: Any) -> Any:
        try:
            bus.emit(StatusEvent(tool=f"mcp:{server}", message=str(getattr(params, "data", ""))))
        except Exception:  # noqa: BLE001 - status is best-effort; never break the handler
            pass
        return await original(params)

    return logging_callback


def _wrap_message(original: Callable, bus: StatusBus, server: str,
                  token_map: dict[Any, str]) -> Callable:
    async def message_handler(message: Any) -> Any:
        try:
            root = getattr(message, "root", None)
            if _is_progress(root):
                p = root.params
                tool = token_map.get(p.progressToken, f"mcp:{server}")
                bus.emit(StatusEvent(tool=tool, message=p.message or "",
                                     current=p.progress, total=p.total))
        except Exception:  # noqa: BLE001 - best-effort; never break the handler
            pass
        return await original(message)

    return message_handler


def install_status_wrappers(bus: StatusBus, tool: Any, server: str) -> dict[Any, str]:
    """Wrap the tool's logging/message handlers (before connect) to emit StatusEvents.

    Returns a (mutable) token->tool-name map shared with the message wrapper; it is empty
    until ``inject_progress_tokens`` populates it after connect. Missing seams are skipped.
    """
    token_map: dict[Any, str] = {}
    orig_logging = getattr(tool, "logging_callback", None)
    if callable(orig_logging):
        tool.logging_callback = _wrap_logging(orig_logging, bus, server)
    orig_message = getattr(tool, "message_handler", None)
    if callable(orig_message):
        tool.message_handler = _wrap_message(orig_message, bus, server, token_map)
    return token_map


def inject_progress_tokens(tool: Any, server: str, token_map: dict[Any, str]) -> None:
    """After connect: give each tool a stable progressToken so the server emits progress.

    Writes ``progressToken`` into MAF's per-tool ``_tool_call_meta_by_name`` (preserving any
    existing meta) and records token->name in ``token_map``. No-op if the seam is absent.
    """
    meta_by_name = getattr(tool, "_tool_call_meta_by_name", None)
    if meta_by_name is None:
        return
    for fn in getattr(tool, "functions", []):
        name = getattr(fn, "name", None)
        if not name:
            continue
        token = f"sheath:{server}:{name}"
        merged = dict(meta_by_name.get(name) or {})
        merged["progressToken"] = token
        meta_by_name[name] = merged
        token_map[token] = name
