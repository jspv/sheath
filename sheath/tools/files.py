"""File tools, all confined to the session root via safe_path.

Every tool holds the session's io_lock (via ``Session.file_io``, which also refuses while
the sandbox is quarantined), so no sandboxed code runs between a path check and its use, and reads only regular files with bounded memory.
"""

from __future__ import annotations

import errno
import os
import re
import stat
import tempfile
from pathlib import Path

from .. import workspace
from ..paths import safe_path
from ..session import Session

_DEFAULT_LIMIT = 2000
_MAX_READ_CHARS = 50_000  # hard cap so a single huge line (e.g. minified HTML) can't flood context
_BINARY_PROBE_BYTES = 8192
_MAX_LISTED = 2000
_MAX_EDIT_BYTES = 10 * 1024 * 1024  # edit_file holds the whole file in memory
_EDIT_PREVIEW_LINES = 3             # context lines either side of an edit in the reply
_EDIT_PREVIEW_CHARS = 4000
_MAX_REPORTED_LINES = 5             # line numbers named when `old` is ambiguous
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")  # read_file's line breaks


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


def read_file(session: Session, path: str, start_line: int = 1, limit: int = _DEFAULT_LIMIT,
              max_chars: int = _MAX_READ_CHARS, line_numbers: bool = False) -> str:
    """Read up to ``limit`` lines starting at 1-based line ``start_line`` from a file under
    the root. Line numbers match ``search`` hits, so a hit's ``line`` can be passed straight in.
    With ``line_numbers``, each line is prefixed with its number and a tab, like ``cat -n``.

    The result is capped at ``max_chars`` characters, and the file is streamed line by line
    with a per-line bound, so neither a huge file nor one enormous line is ever held whole.
    Only regular files are read (a FIFO or device would block or never end); a binary file
    is refused with a pointer to ``run_python``.
    """
    if start_line < 1:
        raise ValueError("start_line is 1-based and must be at least 1")
    if limit < 0:
        raise ValueError("limit must be non-negative")
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
            for lineno, line in enumerate(workspace.iter_bounded_lines(f, max_chars + 1),
                                          start=1):
                if lineno < start_line:
                    continue
                if lineno >= start_line + limit:
                    break
                if line_numbers:
                    line = f"{lineno:>6}\t{line}"
                pieces.append(line)
                size += len(line)
                if size > max_chars:
                    truncated = True
                    break
    text = "".join(pieces)
    if truncated:
        text = (text[:max_chars]
                + f"\n... [truncated at {max_chars} chars; narrow the read with start_line/limit "
                  "or use search to locate what you need]")
    return text


def write_file(session: Session, path: str, content: str) -> str:
    """Write ``content`` to a file under the root, creating parent dirs. Returns a confirmation.

    Paths that resolve into a sheath-managed directory (handles/, inputs/, .scripts/,
    .sheath/) are refused: those hold state the harness vouches for.
    """
    with session.file_io():
        target = safe_path(session.root, path)
        if workspace.is_protected(session.root, target):
            raise ValueError(f"{path!r} is in a directory managed by sheath; write elsewhere "
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


def list_files(session: Session, path: str = ".", pattern: str | None = None) -> list[str]:
    """List regular files (recursively) under ``path`` as root-relative POSIX paths.

    ``pattern`` is a glob matched against each path relative to ``path``: ``*`` stays within
    one directory and ``**`` spans any number (``**/*.csv``, ``data/*.json``). Links and
    sheath's private state directory are skipped. At most ``_MAX_LISTED`` paths are returned,
    followed by a note saying how many were left out.
    """
    matcher = _compile_glob(pattern) if pattern else None
    with session.file_io():
        base = safe_path(session.root, path or ".")  # "" / None -> the root
        if base.is_file():
            rel = base.relative_to(session.root).as_posix()
            return [rel] if matcher is None or matcher.fullmatch(base.name) else []
        out: list[str] = []
        extra = 0
        for p in sorted(base.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(session.root)
            if rel.parts[0] == ".sheath":
                continue
            if matcher is not None and not matcher.fullmatch(p.relative_to(base).as_posix()):
                continue
            if len(out) < _MAX_LISTED:
                out.append(rel.as_posix())
            else:
                extra += 1
    if extra:
        out.append(f"... [{extra} more files not shown; list a subdirectory or narrow the "
                   "pattern]")
    return out


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a relative glob into a regex over POSIX relative paths.

    ``**/`` matches zero or more directories, ``**`` anything, ``*`` and ``?`` stay within one
    path segment, and ``[...]`` is a character class. Matching is done on paths from the walk,
    so a pattern can only ever narrow what ``list_files`` would list anyway; the checks below
    just turn nonsense into a clear error.
    """
    if pattern.startswith(("/", "\\")) or ".." in re.split(r"[/\\]", pattern):
        raise ValueError(f"invalid pattern {pattern!r}: must be relative and contain no '..'")
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[" and (end := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1:end]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out), re.DOTALL)


def edit_file(session: Session, path: str, old: str, new: str,
              replace_all: bool = False) -> str:
    """Replace the exact text ``old`` with ``new`` in a file under the root.

    ``old`` must occur exactly once unless ``replace_all`` is set; zero or several matches
    are an error that says where to look, and the file is left untouched. The file keeps its
    own line endings and permissions, and is replaced atomically (temp file + rename), so a
    failure never leaves it half-written. The same directories ``write_file`` refuses are
    refused here. Returns the replacement count and the edited lines, numbered.
    """
    if not old:
        raise ValueError("old must be non-empty; use write_file to create or overwrite a file")
    if old == new:
        raise ValueError("old and new are identical; nothing to change")
    with session.file_io():
        target = safe_path(session.root, path)
        if workspace.is_protected(session.root, target):
            raise ValueError(f"{path!r} is in a directory managed by sheath; edit files "
                             "elsewhere (e.g. under outputs/)")
        fd = _open_checked(target, os.O_RDONLY, path)
        try:
            st = os.fstat(fd)
            if st.st_size > _MAX_EDIT_BYTES:
                raise ValueError(f"{path!r} is {st.st_size} bytes; edit_file handles files up "
                                 f"to {_MAX_EDIT_BYTES} bytes -- use run_python for larger ones")
            data = b""
            while chunk := os.read(fd, 1 << 20):
                data += chunk
                if len(data) > _MAX_EDIT_BYTES:   # grew after the fstat
                    raise ValueError(f"{path!r} is larger than {_MAX_EDIT_BYTES} bytes")
        finally:
            os.close(fd)
        if b"\0" in data[:_BINARY_PROBE_BYTES]:
            raise ValueError(f"{path!r} is a binary file; edit it inside run_python instead")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{path!r} is not valid UTF-8; edit it inside run_python "
                             "instead") from e

        count = text.count(old)
        if count == 0:
            raise ValueError(_no_match_message(path, text, old))
        if count > 1 and not replace_all:
            shown = ", ".join(map(str, _match_lines(text, old, _MAX_REPORTED_LINES)))
            raise ValueError(f"old occurs {count} times in {path!r} (lines {shown}"
                             f"{', ...' if count > _MAX_REPORTED_LINES else ''}); add "
                             "surrounding text to make it unique, or set replace_all=True")
        first = text.index(old)
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        _replace_atomically(target, updated.encode("utf-8"), st.st_mode)
        rel = target.relative_to(session.root).as_posix()
    return (f"replaced {count if replace_all else 1} occurrence(s) in {rel}\n"
            + _edit_preview(updated, first, len(new)))


def _line_of(text: str, index: int) -> int:
    """1-based line number of ``index`` in ``text``, counting lines as read_file does."""
    head = text[:index]
    return head.count("\n") + head.count("\r") - head.count("\r\n") + 1


def _match_lines(text: str, old: str, limit: int) -> list[int]:
    """The first ``limit`` distinct line numbers holding ``old``. Only these are located: a
    file can hold millions of matches, and each lookup scans from the top."""
    lines: list[int] = []
    start = 0
    while len(lines) < limit and (i := text.find(old, start)) != -1:
        n = _line_of(text, i)
        if not lines or lines[-1] != n:
            lines.append(n)
        start = i + len(old)
        breaks = [b for b in (text.find("\n", start), text.find("\r", start)) if b != -1]
        if not breaks:
            break                      # no later line can hold another match
        start = min(breaks)            # skip the rest of this line
    return lines


def _no_match_message(path: str, text: str, old: str) -> str:
    """Explain a miss, pointing at a near match when only case or whitespace differs."""
    msg = f"old was not found in {path!r}"
    folded = text.casefold().find(old.casefold())
    if folded != -1 and len(text.casefold()) == len(text):   # indices line up
        return f"{msg}; a case-insensitive match starts on line {_line_of(text, folded)}"
    words = old.split()
    if words:
        loose = r"\s+".join(map(re.escape, words))
        m = re.search(loose, text)
        if m:
            return (f"{msg}; a match differing only in whitespace starts on line "
                    f"{_line_of(text, m.start())} -- read it with line_numbers=True and copy "
                    "the text exactly")
    return f"{msg}; read the file and copy the text to replace exactly, including whitespace"


def _edit_preview(text: str, start: int, length: int) -> str:
    """The lines around ``text[start:start + length]``, numbered and bounded."""
    lines = _LINE_RE.findall(text)   # not splitlines(): it also breaks on \f, \x1c, ...
    first = _line_of(text, start)
    last = _line_of(text, start + max(length - 1, 0))
    lo = max(1, first - _EDIT_PREVIEW_LINES)
    hi = min(len(lines), last + _EDIT_PREVIEW_LINES)
    out = "".join(f"{n:>6}\t{lines[n - 1]}" for n in range(lo, hi + 1))
    if len(out) > _EDIT_PREVIEW_CHARS:
        out = out[:_EDIT_PREVIEW_CHARS] + "\n... [preview truncated]"
    return out


def _replace_atomically(target: Path, data: bytes, mode: int) -> None:
    """Write ``data`` beside ``target`` and rename it into place, keeping ``mode``."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, stat.S_IMODE(mode))
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, target)
    except BaseException:
        if fd != -1:
            os.close(fd)
        os.unlink(tmp)
        raise
