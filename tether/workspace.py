"""Workspace trust zones: which parts of a session root the parent owns, and the only way
the parent reads what sandboxed code wrote.

The root is bind-mounted read-write into the sandbox, so anything directly under it may be
a link sandboxed code planted. Four *protected* directories are created by the parent
before any sandboxed code runs and are mounted read-only into the container, so sandboxed
code can neither write inside them nor replace them. Sandbox-to-parent data travels only
through a per-run directory under ``.tether/runs/``, and is read here with no-follow,
regular-file-only, size-capped opens.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator
from pathlib import Path

from .paths import validate_segment

PROTECTED_DIRS: tuple[str, ...] = ("handles", "inputs", ".scripts", ".tether")
_STATE = ".tether"
_STATE_SUBDIRS: tuple[str, ...] = ("tmp", "runs")


class WorkspaceTamperedError(RuntimeError):
    """A protected directory is missing or is not a real directory. Not recoverable: the
    parent would otherwise write trusted state wherever a link points."""


class ControlFileError(ValueError):
    """A sandbox-written control file was rejected (not a regular file, or too large)."""


def _is_real_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _require_real_dir(path: Path, shown: str, root: Path) -> None:
    if not _is_real_dir(path):
        raise WorkspaceTamperedError(
            f"workspace directory {shown!r} under {root} is missing or is not a real "
            f"directory; refusing to use this workspace")


def verify_layout(root: Path | str) -> None:
    """Raise unless every protected directory, and the parent's own ``.tether/tmp`` and
    ``.tether/runs``, is a real directory (never follows a link)."""
    root = Path(root)
    for name in PROTECTED_DIRS:
        _require_real_dir(root / name, name, root)
    for sub in _STATE_SUBDIRS:
        _require_real_dir(root / _STATE / sub, f"{_STATE}/{sub}", root)


def ensure_layout(root: Path | str) -> None:
    """Create the protected directories (and ``.tether/tmp``, ``.tether/runs``); refuse a
    root where one exists as anything else."""
    root = Path(root)
    for name in PROTECTED_DIRS:
        path = root / name
        if not os.path.lexists(path):
            path.mkdir()
    if _is_real_dir(root / _STATE):
        for sub in _STATE_SUBDIRS:
            path = root / _STATE / sub
            if not os.path.lexists(path):
                path.mkdir()          # no parents: .tether was just proven a real directory
    verify_layout(root)


def is_protected(root: Path | str, resolved: Path) -> bool:
    """Whether an already-resolved path lies in (or is) a protected directory of ``root``.

    Decides by filesystem identity (st_dev, st_ino) rather than spelling, to catch
    case-variant paths on case-insensitive filesystems.
    """
    root = Path(root).resolve()
    resolved = Path(resolved)

    # Try the cheap path check first
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return False

    if not parts:
        return False

    # Cheap check: if the first component matches by name, it's protected
    if parts[0] in PROTECTED_DIRS:
        return True

    # Expensive check: if the first component is the same inode as a protected dir,
    # it's protected (handles case-insensitive filesystems)
    first_component = root / parts[0]

    # If the path doesn't exist, it's not protected
    try:
        first_st = os.lstat(first_component)
    except OSError:
        return False

    # Compare inode identity against each protected directory
    for protected_name in PROTECTED_DIRS:
        try:
            protected_st = os.lstat(root / protected_name)
            if (first_st.st_dev == protected_st.st_dev and
                first_st.st_ino == protected_st.st_ino):
                return True
        except OSError:
            continue

    return False


def state_dir(root: Path | str) -> Path:
    return Path(root).resolve() / _STATE


def _state_subdir(root: Path | str, sub: str) -> Path:
    """``.tether/<sub>``, created if missing, refused if it is a link or not a directory.

    Never ``mkdir(parents=True, exist_ok=True)``: that would follow a planted link at either
    level and put parent-owned files wherever it points.
    """
    root = Path(root).resolve()
    state = root / _STATE
    _require_real_dir(state, _STATE, root)
    path = state / sub
    if not os.path.lexists(path):
        path.mkdir()
    _require_real_dir(path, f"{_STATE}/{sub}", root)
    return path


def scratch_dir(root: Path | str) -> Path:
    """Parent-only scratch space (downloads, snapshots). Created on demand."""
    return _state_subdir(root, "tmp")


def new_run_dir(root: Path | str) -> Path:
    """Create and return a fresh per-run exchange directory under ``.tether/runs/``."""
    path = _state_subdir(root, "runs") / secrets.token_hex(8)
    path.mkdir()
    return path


def read_control_file(run_dir: Path, name: str, max_bytes: int) -> bytes | None:
    """Read a sandbox-written file from a run directory. ``None`` if absent.

    Never follows a link, never blocks on a FIFO, never reads more than ``max_bytes``. The
    size is taken from the open descriptor, so it describes the bytes actually read.
    """
    validate_segment(name, what="control file name")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(Path(run_dir) / name, flags)
    except FileNotFoundError:
        return None
    except OSError as e:          # ELOOP for a symlink, ENXIO for some special files
        raise ControlFileError("not a regular file") from e

    # Check if it's a regular file immediately, before fdopen
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise ControlFileError("not a regular file")
    if st.st_size > max_bytes:
        os.close(fd)
        raise ControlFileError(f"too large ({st.st_size} bytes > {max_bytes})")

    with os.fdopen(fd, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ControlFileError(f"too large (more than {max_bytes} bytes)")
    return data


def require_regular(path: Path) -> Path:
    """Return ``path`` if it is a regular file; raise ``ValueError`` for anything else that
    exists (directory, FIFO, device). A missing path raises ``FileNotFoundError``."""
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise ValueError(f"not a regular file: {path.name}")
    return path


def iter_bounded_lines(f, max_line_chars: int, chunk: int = 65536) -> Iterator[str]:
    """Yield the lines of text stream ``f``, each cut to ``max_line_chars``.

    A line longer than the bound is consumed in ``chunk``-sized reads and discarded past
    the bound, so one enormous line costs bounded memory and still counts as one line.
    """
    while True:
        piece = f.readline(chunk)
        if not piece:
            return
        kept = piece[:max_line_chars]
        # newline="" streams end a line at "\n", "\r\n" or a bare "\r".
        while piece[-1] not in "\r\n":
            piece = f.readline(chunk)
            if not piece:
                break
            room = max_line_chars - len(kept)
            if room > 0:
                kept += piece[:room]
        yield kept
