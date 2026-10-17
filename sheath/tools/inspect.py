"""inspect_handle: on-demand deeper look at a stored handle (more preview / stats).

Everything here is bounded and none of it materializes a handle: text and JSON are read
with a capped read, a dataframe contributes only its first record batch and its footer
statistics. Handle bytes may have been written by sandboxed code.
"""

from __future__ import annotations

import json
from typing import Any

from ..handles import first_group_too_large
from ..session import Session

_MAX_ROWS = 100
_MAX_INSPECT_CHARS = 20_000
_MAX_STATS_COLS = 200
_MAX_SCALAR_CHARS = 200


def inspect_handle(session: Session, handle_id: str, rows: int = 20,
                   stats: bool = False) -> dict:
    """Return a deeper view of a handle than its summary: up to ``rows`` (1..100) preview
    lines for text/json, or head rows for a dataframe. ``stats=True`` adds per-column
    min/max/null_count for a dataframe, taken from the file's footer statistics."""
    handle = session.store.manifest_handles()[handle_id]  # raises KeyError if unknown
    path = session.store.verify(handle_id)                # digest check, no load
    rows = max(1, min(int(rows), _MAX_ROWS))
    out = handle.summary()
    if handle.kind == "dataframe":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        too_large = first_group_too_large(pf)
        if too_large is not None:
            # Arrow decodes the whole first row group regardless of batch size.
            head = []
            out["head_omitted"] = f"first row group is {too_large} bytes; inspect it in run_python"
            batch = None
        else:
            batch = next(pf.iter_batches(batch_size=rows), None)
            head = batch.to_pandas().to_dict(orient="records") if batch is not None else []
        kept = len(head)
        while kept > 1 and len(json.dumps(head[:kept], default=str)) > _MAX_INSPECT_CHARS:
            kept //= 2
        if kept < len(head):
            out["head_truncated"] = True
        head = head[:kept]
        if len(json.dumps(head, default=str)) > _MAX_INSPECT_CHARS:
            # A single row is still too large: show it as clipped text, not as data.
            head = [{"row_text": json.dumps(head[0], default=str)[:_MAX_INSPECT_CHARS - 100]}]
            out["head_truncated"] = True
        out["head"] = head
        if stats:
            out["stats"] = _footer_stats(pf)
    elif handle.kind == "binary":
        pass                                              # the summary preview is all there is
    else:
        with path.open(encoding="utf-8", errors="replace") as f:
            text = f.read(_MAX_INSPECT_CHARS)
        out["preview"] = "\n".join(text.splitlines()[:rows])
    return out


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_MAX_SCALAR_CHARS]


def _footer_stats(pf) -> dict[str, dict[str, Any]]:
    """Per-column min / max / null_count aggregated over row groups, from footer metadata.

    No data page is read. ``min``/``max`` are reported only when every non-empty row group
    carries them (an all-null group contributes nothing) and they are comparable; otherwise
    they are ``None`` rather than a partial answer. ``null_count`` is ``None`` when any
    non-empty row group lacks it.
    """
    md = pf.metadata
    out: dict[str, dict[str, Any]] = {}
    for j in range(min(md.num_columns, _MAX_STATS_COLS)):
        name = str(md.schema.column(j).path)[:128]
        lo = hi = None
        known = True                       # min/max still trustworthy (sticky once False)
        nulls: int | None = 0
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            if rg.num_rows == 0:
                continue
            st = rg.column(j).statistics
            if st is None:
                known = False
                nulls = None
                continue
            if st.null_count is None:
                nulls = None
            elif nulls is not None:
                nulls += st.null_count
            if not known:
                continue
            if st.has_min_max:
                try:
                    lo = st.min if lo is None or st.min < lo else lo
                    hi = st.max if hi is None or st.max > hi else hi
                except TypeError:          # incomparable across row groups
                    known = False
            elif st.null_count != rg.num_rows:   # not merely all-null: stats are missing
                known = False
        if not known:
            lo = hi = None
        out[name] = {"min": _scalar(lo), "max": _scalar(hi), "null_count": nulls}
    return out
