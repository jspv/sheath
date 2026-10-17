"""Injected into the sandbox subprocess as the top-level module ``sheath_sandbox``.

Communicates with the parent sheath only via env vars and a per-run exchange directory:
  SHEATH_ROOT             session root directory (handles/ and inputs/ are read-only here)
  SHEATH_RUN_DIR          this run's exchange directory; the only place to hand data back
  SHEATH_PUBLISH_ENABLED  "1" when the host accepts publish() requests

Files in SHEATH_RUN_DIR:
  registry.json      { handle_id: {kind, path} } for existing handles (read)
  new_handles.jsonl  one record per save() (appended)
  emit.json          the emit() payload (written)
  publish.jsonl      one record per publish() (appended)
  save_<n><ext>      save() payloads; the parent moves them into handles/
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ROOT = Path(os.environ["SHEATH_ROOT"])
_RUN = Path(os.environ["SHEATH_RUN_DIR"])
_REGISTRY = json.loads((_RUN / "registry.json").read_text(encoding="utf-8"))
_NEW = _RUN / "new_handles.jsonl"
_EMIT = _RUN / "emit.json"
_PUBLISH = _RUN / "publish.jsonl"
_SAVED_THIS_RUN: set[str] = set()  # ids written by save() in this process
# Mirrors sheath.handles.HANDLE_ID_RE. Ergonomics only: the parent enforces it for real.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def load(handle_id: str) -> Any:
    meta = _REGISTRY[handle_id]
    path = _ROOT / meta["path"]
    kind = meta["kind"]
    if kind == "dataframe":
        import pandas as pd
        return pd.read_parquet(path)
    if kind == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if kind == "binary":
        return str(path)  # binary file: hand back the path to open with pandas/Docling/etc.
    return path.read_text(encoding="utf-8")


def save(handle_id: str, obj: Any, source: str = "run_python") -> str:
    """Write ``obj`` into the run directory and tell the parent it exists.

    Only id/kind/file/source are reported: the parent moves the file into handles/ and
    derives every field that reaches model context from its bytes, so there is nothing to
    keep in sync here and nothing this side can misreport.
    """
    import pandas as pd

    # Ergonomics for honest code only, NOT a security control: a hostile child can skip
    # save() and write the control file directly. The parent's checks are the boundary.
    if not isinstance(handle_id, str) or not _ID_RE.match(handle_id):
        raise ValueError(
            f"invalid handle id {handle_id!r}: use letters, digits, '_', '-', '.', "
            f"starting with a letter or digit, at most 64 characters")
    if handle_id in _REGISTRY or handle_id in _SAVED_THIS_RUN:
        raise ValueError(
            f"handle id {handle_id!r} already exists and handles are immutable; "
            f"save under a new id"
        )

    n = len(_SAVED_THIS_RUN) + 1
    if isinstance(obj, pd.DataFrame):
        kind, name = "dataframe", f"save_{n}.parquet"
        obj.to_parquet(_RUN / name)
    elif isinstance(obj, (dict, list)):
        kind, name = "json", f"save_{n}.json"
        (_RUN / name).write_text(json.dumps(obj, default=str), encoding="utf-8")
    else:
        kind, name = "text", f"save_{n}.txt"
        (_RUN / name).write_text(str(obj), encoding="utf-8")

    with _NEW.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": handle_id, "kind": kind, "file": name,
                            "source": source}) + "\n")
    _SAVED_THIS_RUN.add(handle_id)
    return handle_id


def emit(obj: Any) -> None:
    _EMIT.write_text(json.dumps(obj, default=str), encoding="utf-8")


def publish(path: str, name: str | None = None, description: str | None = None) -> None:
    """Ask the host to deliver a finished workspace file to the user.

    The request is recorded here and processed by the parent after this script exits cleanly
    (the parent re-validates it; this side is not trusted). Raises if the host has not enabled
    publishing for this session.
    """
    if os.environ.get("SHEATH_PUBLISH_ENABLED") != "1":
        raise RuntimeError("publishing is not enabled for this session "
                           "(the host has not configured a publish callback)")
    p = os.path.abspath(os.fspath(path))
    root = os.path.abspath(str(_ROOT))
    if os.path.commonpath([p, root]) == root:
        p = os.path.relpath(p, root)   # convenience: /workspace/x -> x (not a trust step)
    with _PUBLISH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"path": p, "name": name, "description": description}) + "\n")
