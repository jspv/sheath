"""Spill: turn oversized/structured tool results into handles via MAF result_parser.

A plugged-in capability (a plain tool function or an MCP server tool) returns its raw
Python value; MAF calls our ``result_parser`` on it *before serialization*. We write an
oversized/structured value to the handle store and return the lightweight handle summary
the model sees; small values defer to MAF's default parsing. The sheath's own built-in
tools are NOT given this parser -- they already manage their own output.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent_framework import FunctionTool
from agent_framework._types import Content

from .session import Session
from .status import report_progress


def _is_handle_summary(obj: Any) -> bool:
    return isinstance(obj, dict) and {"id", "kind", "path"} <= obj.keys()


def normalize_mcp_result(result: Any) -> Any:
    """Collapse an MCP ``list[Content]`` return into its underlying payload.

    MAF hands our parser the raw return value, and MCP tools return their result as a
    ``list[Content]`` -- almost always a single text item carrying JSON. Left as-is, a
    spill would store the Content objects (double-encoded as ``["[{...}]"]``) instead of
    the data. We join the text, parse it as JSON when possible (so it stores as a clean
    ``json`` handle), and otherwise return the raw string.

    Any non-text Content (image/data/resource has ``text is None``) means a plain join
    would silently drop bytes, so such a result is returned unchanged for passthrough.
    """
    if not (isinstance(result, list) and result):
        return result
    if not all(isinstance(c, Content) and getattr(c, "text", None) for c in result):
        return result
    text = "".join(c.text for c in result)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return text
    return parsed if isinstance(parsed, (dict, list)) else text


class SpillLimitExceeded(RuntimeError):
    """A tool returned more than ``config.max_spill_bytes`` -- rejected, not stored.

    The upper edge of the spill-over zone: spilling is lossless, so we never truncate,
    but we also won't silently absorb an unbounded dump from a possibly-untrusted tool
    or MCP server. Failing loudly keeps the disk/cost surface bounded and points the
    developer at the fix (paginate/filter, or raise the cap).
    """

    def __init__(self, tool_name: str, size: int, limit: int) -> None:
        self.tool_name = tool_name
        self.size = size
        self.limit = limit
        super().__init__(
            f"tool {tool_name!r} returned {size} bytes, over the {limit}-byte spill cap "
            f"(config.max_spill_bytes). Have the source paginate or filter its results, "
            f"or raise max_spill_bytes if this volume is expected."
        )


def _is_dataframe(result: Any) -> bool:
    try:
        import pandas as pd
        return isinstance(result, pd.DataFrame)
    except ImportError:
        return False


def _spill_size_bytes(result: Any) -> int | None:
    """Bytes a result would occupy as a handle, or ``None`` if it isn't a spillable kind.

    A handle summary returns ``None`` so it never re-spills. The same measurement drives
    both edges of the zone (threshold and cap), so the two decisions can't drift apart.
    """
    if _is_dataframe(result):
        return int(result.memory_usage(deep=True).sum())
    if isinstance(result, (bytes, bytearray)):
        return len(result)
    if isinstance(result, str):
        return len(result.encode())
    if isinstance(result, (dict, list)):
        if _is_handle_summary(result):
            return None
        return len(json.dumps(result, default=str).encode())
    return None


def _should_spill(result: Any, threshold_bytes: int) -> bool:
    # DataFrames always become handles (they exist to be analyzed in the sandbox);
    # other kinds spill only once they cross the lower threshold.
    if _is_dataframe(result):
        return True
    size = _spill_size_bytes(result)
    return size is not None and size > threshold_bytes


def _maybe_spill(session: Session, tool_name: str, result: Any) -> Any:
    result = normalize_mcp_result(result)
    if not _should_spill(result, session.config.spill_threshold_bytes):
        return result
    size = _spill_size_bytes(result) or 0
    if size > session.config.max_spill_bytes:
        raise SpillLimitExceeded(tool_name, size, session.config.max_spill_bytes)
    handle = session.store.put(result, source=f"tool:{tool_name}")
    report_progress(f"stored {size} bytes as {handle.id}", tool=tool_name)
    return handle.summary()


def make_spill_parser(session: Session, tool_name: str) -> Callable[[Any], list[Content]]:
    """A MAF ``result_parser``: spill oversized returns, else default-parse."""
    def parse(result: Any) -> list[Content]:
        return FunctionTool.parse_result(_maybe_spill(session, tool_name, result))
    return parse


def spill_tool(session: Session, fn: Callable) -> FunctionTool:
    """Wrap a plain developer callable as a FunctionTool whose return is spilled."""
    name = getattr(fn, "__name__", "tool")
    return FunctionTool(
        func=fn,
        name=name,
        description=(fn.__doc__ or "").strip(),
        result_parser=make_spill_parser(session, name),
    )


def looks_like_mcp(tool: Any) -> bool:
    """Duck-typed MCP detection: connectable, closeable, exposes ``.functions``.

    Avoids coupling to MAF's MCP class names and lets tests use a fake. Plain callables
    have no ``.functions`` attribute, so there are no false positives.
    """
    return (
        callable(getattr(tool, "connect", None))
        and callable(getattr(tool, "close", None))
        and hasattr(tool, "functions")
    )
