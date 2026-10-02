"""File tools, all confined to the session root via safe_path.

Every tool holds the session's io_lock, so no sandboxed code runs between a path check and
its use, and reads only regular files with bounded memory.
"""

from __future__ import annotations

from .. import workspace
from ..paths import safe_path
from ..session import Session

_DEFAULT_LIMIT = 2000
_MAX_READ_CHARS = 50_000  # hard cap so a single huge line (e.g. minified HTML) can't flood context
_BINARY_PROBE_BYTES = 8192
_MAX_LISTED = 2000


def read_file(session: Session, path: str, offset: int = 0, limit: int = _DEFAULT_LIMIT,
              max_chars: int = _MAX_READ_CHARS) -> str:
    """Read up to ``limit`` lines starting at line ``offset`` from a file under the root.

    The result is capped at ``max_chars`` characters, and the file is streamed line by line
    with a per-line bound, so neither a huge file nor one enormous line is ever held whole.
    Only regular files are read (a FIFO or device would block or never end); a binary file
    is refused with a pointer to ``run_python``.
    """
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    with session.io_lock:
        target = workspace.require_regular(safe_path(session.root, path))
        with target.open("rb") as probe:
            if b"\0" in probe.read(_BINARY_PROBE_BYTES):
                raise ValueError(f"{path!r} is a binary file; open it inside run_python "
                                 "instead of reading it as text")
        pieces: list[str] = []
        size = 0
        truncated = False
        with target.open(encoding="utf-8", errors="replace", newline="") as f:
            for lineno, line in enumerate(workspace.iter_bounded_lines(f, max_chars + 1)):
                if lineno < offset:
                    continue
                if lineno >= offset + limit:
                    break
                pieces.append(line)
                size += len(line)
                if size > max_chars:
                    truncated = True
                    break
    text = "".join(pieces)
    if truncated:
        text = (text[:max_chars]
                + f"\n... [truncated at {max_chars} chars; narrow the read with offset/limit "
                  "or use search to locate what you need]")
    return text


def write_file(session: Session, path: str, content: str) -> str:
    """Write ``content`` to a file under the root, creating parent dirs. Returns a confirmation.

    Paths that resolve into a tether-managed directory (handles/, inputs/, .scripts/,
    .tether/) are refused: those hold state the harness vouches for.
    """
    with session.io_lock:
        target = safe_path(session.root, path)
        if workspace.is_protected(session.root, target):
            raise ValueError(f"{path!r} is in a directory managed by tether; write elsewhere "
                             "(e.g. under outputs/) and use save() in run_python for handles")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            workspace.require_regular(target)
        target.write_text(content, encoding="utf-8")
        rel = target.relative_to(session.root).as_posix()
    return f"wrote {len(content.encode())} bytes to {rel}"


def list_files(session: Session, path: str = ".") -> list[str]:
    """List regular files (recursively) under ``path`` as root-relative POSIX paths.

    Links and tether's private state directory are skipped. At most ``_MAX_LISTED`` paths
    are returned, followed by a note saying how many were left out.
    """
    with session.io_lock:
        base = safe_path(session.root, path or ".")  # "" / None -> the root
        if base.is_file():
            return [base.relative_to(session.root).as_posix()]
        out: list[str] = []
        extra = 0
        for p in sorted(base.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(session.root)
            if rel.parts[0] == ".tether":
                continue
            if len(out) < _MAX_LISTED:
                out.append(rel.as_posix())
            else:
                extra += 1
    if extra:
        out.append(f"... [{extra} more files not shown; list a subdirectory]")
    return out
