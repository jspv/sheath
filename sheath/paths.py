"""Single chokepoint that confines every model-supplied path to the session root."""

from __future__ import annotations

import unicodedata
from pathlib import Path

_MAX_SEGMENT = 255
_MAX_FILENAME = 128


class PathEscapesRootError(ValueError):
    """Raised when a candidate path resolves outside the session root."""


def safe_path(root: Path | str, candidate: Path | str) -> Path:
    """Resolve ``candidate`` against ``root`` and guarantee it stays inside it.

    ``.resolve()`` normalizes ``..`` and follows symlinks, so a symlink inside
    the root that points outside is caught here rather than exploited.

    An empty/whitespace-only ``candidate`` is rejected: a file tool receiving it
    signals an upstream bug, not a request for the root. Note: a symlink *loop*
    inside the root is not rejected (it stays inside root, so it is not an escape);
    the OS raises ``OSError`` (ELOOP) when the returned path is later opened.
    """
    if not str(candidate).strip():
        raise PathEscapesRootError("empty candidate path")
    root_resolved = Path(root).resolve()
    p = Path(candidate)
    if not p.is_absolute():
        p = root_resolved / p
    resolved = p.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathEscapesRootError(f"path escapes root: {candidate!r}")
    return resolved


def validate_segment(value: str, *, what: str) -> str:
    """Validate an untrusted id that will name exactly one directory under a trusted base.

    Rejects empty, ``.``/``..``, any separator, control characters, and over-long values, so the
    id can neither nest under a sibling nor escape the base. Callers still join the result with
    ``safe_path`` as a containment backstop.
    """
    if (not isinstance(value, str) or value in ("", ".", "..") or "/" in value or "\\" in value
            or len(value) > _MAX_SEGMENT
            or any(unicodedata.category(c).startswith("C") for c in value)):
        raise ValueError(f"invalid {what} (must be a single path segment): {value!r}")
    return value


def _clean_name(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(c for c in base if not unicodedata.category(c).startswith("C")).strip()
    if base in ("", ".", ".."):
        return ""
    if len(base) > _MAX_FILENAME:
        stem, dot, ext = base.rpartition(".")
        if dot and stem and len(ext) < 16:
            base = stem[: _MAX_FILENAME - len(ext) - 1] + "." + ext
        else:
            base = base[:_MAX_FILENAME]
    return base


def safe_filename(name: str | None, *, fallback: str) -> str:
    """Reduce an untrusted display name to a safe basename.

    Keeps only the last path segment (``/`` or ``\\``), strips control/format characters
    (including bidi overrides), trims whitespace, and caps the length while keeping the
    extension. An empty or dot-only result falls back to ``fallback`` (sanitized the same way),
    then to ``"file"``.
    """
    for candidate in (name, fallback):
        if isinstance(candidate, str):
            cleaned = _clean_name(candidate)
            if cleaned:
                return cleaned
    return "file"
