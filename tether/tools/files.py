"""File tools, all confined to the session root via safe_path.

Every tool holds the session's io_lock (via ``Session.file_io``, which also refuses while
the sandbox is quarantined), so no sandboxed code runs between a path check and its use, and reads only regular files with bounded memory.
"""

from __future__ import annotations

import errno
import os
import stat

from .. import workspace
from ..paths import safe_path
from ..session import Session

_DEFAULT_LIMIT = 2000
_MAX_READ_CHARS = 50_000  # hard cap so a single huge line (e.g. minified HTML) can't flood context
_BINARY_PROBE_BYTES = 8192
_MAX_LISTED = 2000


def _open_checked(target, flags: int, shown: str) -> int:
    """Open ``target`` once, never following a final symlink and never blocking on a FIFO,
    and confirm on the descriptor itself that it is a regular file."""
    try:
        fd = os.open(target, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ValueError(f"{shown!r} is a symlink; refusing to follow it") from e
        if e.errno in (errno.EISDIR, errno.ENXIO, errno.ENODEV):
            raise ValueError(f"not a regular file: {shown!r}") from e
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"not a regular file: {shown!r}")
    return fd


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
    with session.file_io():
        target = safe_path(session.root, path)
        fd = _open_checked(target, os.O_RDONLY, path)
        try:
            if b"\0" in os.read(fd, _BINARY_PROBE_BYTES):
                raise ValueError(f"{path!r} is a binary file; open it inside run_python "
                                 "instead of reading it as text")
            os.lseek(fd, 0, os.SEEK_SET)
            f = open(fd, encoding="utf-8", errors="replace", newline="")  # noqa: SIM115
        except BaseException:
            os.close(fd)
            raise
        pieces: list[str] = []
        size = 0
        truncated = False
        with f:
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
    with session.file_io():
        target = safe_path(session.root, path)
        if workspace.is_protected(session.root, target):
            raise ValueError(f"{path!r} is in a directory managed by tether; write elsewhere "
                             "(e.g. under outputs/) and use save() in run_python for handles")
        data = content.encode("utf-8")   # before truncating: a bad string must not empty it
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_checked(target, os.O_WRONLY | os.O_CREAT, path)
        try:
            os.ftruncate(fd, 0)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        rel = target.relative_to(session.root).as_posix()
    return f"wrote {len(data)} bytes to {rel}"


def list_files(session: Session, path: str = ".") -> list[str]:
    """List regular files (recursively) under ``path`` as root-relative POSIX paths.

    Links and tether's private state directory are skipped. At most ``_MAX_LISTED`` paths
    are returned, followed by a note saying how many were left out.
    """
    with session.file_io():
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
