# Workspace Trust Zones and Bounded I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the container tier a real boundary (no parent filesystem operation can be redirected by sandboxed code), bound every tool result and every parent-side read, make parallel tool calls safe, and fix the conversation/AG-UI lifecycle and packaging defects found in the 2026-10-02 review.

**Architecture:** The session root is split into parent-owned protected directories (`handles/`, `inputs/`, `.scripts/`, `.tether/`), mounted read-only into the container, plus a per-run exchange directory (`.tether/runs/<token>/`, mounted read-write at `/tether-run`) that is the only place the parent reads sandbox-written data from, with no-follow, regular-file, size-capped reads. Handle payloads are renamed from the exchange directory into `handles/` by the parent. Process output, file reads, parquet previews and HTTP bodies are all streamed with caps.

**Tech Stack:** Python 3.12, pytest, ruff, uv, httpx, pandas/pyarrow, podman/docker CLI, Microsoft Agent Framework.

**Spec:** `docs/superpowers/specs/2026-10-02-workspace-trust-zones-design.md` — read it before starting any task.

## Global Constraints

- TDD: write the failing test first for every unit; run it and see it fail before implementing.
- Never name any AI assistant or coding-agent product in tracked files (code, comments, docs, commit messages). Before each commit run the forbidden-reference grep given in the project's local, untracked conventions file ("Hard rules", rule 1) and expect no output. The pattern is not reproduced here because this file is tracked and would match itself.
- Do not push. Work on branch `feat/workspace-trust-zones` (create it from `main` in Task 1). Commit at the end of each task.
- Run commands with `uv run`; sync with `uv sync --prerelease=allow`.
- The test suite pins the sandbox backend to `local` (see `tests/conftest.py`); container tests are gated on a live runtime and must create roots under `Path.home() / ".tether" / "_ctest"`.
- The podman VM is shared. Never run any `prune`, `rm -f $(...)`, `rmi -a`, `kill --all` or `stop --all`. Remove only containers this code created, by exact name.
- `rm` and `cp` are aliased to interactive in this shell: use `command rm -f` / `command cp -f` in any scripted step.
- Protected directory names, exactly: `handles`, `inputs`, `.scripts`, `.tether`.
- Handle id pattern, exactly: `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`.
- Run-directory file names, exactly: `registry.json`, `emit.json`, `new_handles.jsonl`, `publish.jsonl`.
- Environment variables passed to sandboxed code, exactly: `TETHER_ROOT`, `TETHER_RUN_DIR`, `TETHER_PUBLISH_ENABLED`.
- New config defaults, exactly: `SandboxConfig.max_output_bytes = 8192`, `FetchConfig.total_timeout_s = 120.0`, `FetchConfig.max_bytes_ceiling = 100_000_000`.
- Lock order is always `session.io_lock` then the `HandleStore` lock; the store never takes `io_lock`.
- Full-suite command: `uv run pytest -p no:warnings`. Baseline before Task 1: 609 passed, 6 skipped. Lint: `uv run ruff check tether tests`.

## Review Focus

Failure modes the spec implies that are most likely to bite a user; each has a test in the task named.

1. A root created by the previous version (manifest at `handles/_manifest.json`) is reopened — handles must rehydrate and the manifest must move (Task 2, `test_legacy_manifest_is_migrated`).
2. A model script calls `save("eu/sales", df)` or another id outside the pattern — it must fail inside the script with a message that says what a valid id is (Task 4, `test_save_rejects_invalid_id_in_child`).
3. The model passes an enormous `max_bytes` to `fetch_url` — it must be clamped to the ceiling, not honored (Task 10, `test_fetch_url_clamps_model_supplied_max_bytes`).
4. A client disconnects mid-stream from an AG-UI turn — the conversation lock must be released so the next turn is not blocked forever (Task 11, `test_abandoned_agui_stream_releases_the_lock`).
5. `read_file` is pointed at a binary or non-UTF-8 file — it must return a clear "binary file" error, not replacement-character noise or a decode traceback (Task 7, `test_read_file_rejects_binary`).

## Assumptions verified before starting (2026-10-02)

Tested against podman on macOS with the existing sandbox image and exactly the mounts Task 5 builds:

- Inside the container, `write`, `symlink`, `chmod` under `handles/`, `inputs/`, `.scripts/`, `.tether/` fail with "Read-only file system"; `rename` and `rmdir` of each of those directories fail with "Device or resource busy". Existing handle files and the manifest cannot be overwritten or unlinked. The run directory is not writable through its `/workspace/.tether/runs/...` view, only through `/tether-run`.
- Hard-linking a handle file into `/tether-run` or the workspace fails with "Invalid cross-device link".
- Reads of `handles/` and execution of a script in `.scripts/` work through the read-only mounts; ordinary workspace writes still work.
- A payload written in `/tether-run` appears on the host as a regular file owned by the host user; the host can `rename` it into `handles/` and `chmod` it. A symlink planted at `emit.json` is refused by an `O_NOFOLLOW` open.
- `ARG PYTHON_VERSION` before `FROM python:${PYTHON_VERSION}-slim` builds with `--build-arg`.
- ripgrep, when walking a directory, skips symlinks and FIFOs, honors `--max-filesize <bytes>`, and exits 2 on an invalid regex.
- Closing an async generator that holds an `asyncio.Lock` releases it; httpx's mock transport streams a generator body lazily (3 chunks pulled for a 2500-byte cap).

Not verified: **docker** (not installed here). Decided 2026-10-02: podman is docker-compatible, so podman verification is accepted as sufficient; no docker run is required.

Environment note: no standalone `rg` binary is on `PATH` for Python on the development machine (`shutil.which("rg")` is `None`; the interactive shell's `rg` is a function), so the suite exercises only the Python search fallback there. Task 7 adds an rg-only test that skips when `rg` is absent, and Task 13 installs ripgrep in CI so that path is tested.

## File Structure

| File | Responsibility |
|---|---|
| `tether/workspace.py` (new) | Trust-zone layout: protected dirs, run dirs, scratch dir, no-follow capped control-file reads, bounded line iteration |
| `tether/procio.py` (new) | Run a child process with head-and-tail bounded output capture |
| `tether/handles.py` | Thread-safe store, atomic writes, manifest in `.tether/`, `adopt_file`, bounded describers |
| `tether/sandbox.py` | Run-directory protocol, ingestion, local backend |
| `tether/sandbox_container.py` | Read-only protected mounts, `/tether-run`, orphan detection |
| `tether/runtime/tether_sandbox.py`, `tether/runtime/_runner.py` | Child side of the protocol |
| `tether/session.py`, `tether/manager.py` | Root allocation, id validation, artifacts filter |
| `tether/tools/files.py`, `search.py` | Locked, bounded, regular-file-only file tools |
| `tether/tools/code.py`, `inspect.py`, `deliver.py` | Result spill, bounded inspect, in-sandbox conversion |
| `tether/egress.py`, `tools/fetch.py`, `tools/documents.py` | Streaming fetch with deadline and ceiling |
| `tether/conversation.py`, `tether/api.py` | AG-UI turn under the conversation lock; side-effect-free constructor |
| `tether/container_runtime.py`, `tether/runtime/Containerfile` | `PYTHON_VERSION` build argument |
| `.github/workflows/tests.yml` (new), `publish.yml` | CI |
| `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md` | Documentation |

---

### Task 1: `tether/workspace.py` — trust-zone layout and bounded reads

**Files:**
- Create: `tether/workspace.py`
- Test: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `tether.paths.validate_segment(value: str, *, what: str) -> str`
- Produces:
  - `PROTECTED_DIRS: tuple[str, ...] = ("handles", "inputs", ".scripts", ".tether")`
  - `class WorkspaceTamperedError(RuntimeError)`
  - `class ControlFileError(ValueError)`
  - `ensure_layout(root: Path | str) -> None`
  - `verify_layout(root: Path | str) -> None`
  - `is_protected(root: Path | str, resolved: Path) -> bool`
  - `state_dir(root: Path | str) -> Path`
  - `scratch_dir(root: Path | str) -> Path`
  - `new_run_dir(root: Path | str) -> Path`
  - `read_control_file(run_dir: Path, name: str, max_bytes: int) -> bytes | None`
  - `require_regular(path: Path) -> Path`
  - `iter_bounded_lines(f, max_line_chars: int, chunk: int = 65536) -> Iterator[str]`

- [ ] **Step 1: Confirm the branch**

Run: `git branch --show-current`
Expected: `feat/workspace-trust-zones` (already created).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_workspace.py`:

```python
import io
import os

import pytest

from tether import workspace
from tether.workspace import (ControlFileError, WorkspaceTamperedError, ensure_layout,
                              is_protected, iter_bounded_lines, new_run_dir,
                              read_control_file, require_regular, scratch_dir, verify_layout)


def test_ensure_layout_creates_real_protected_dirs(tmp_path):
    ensure_layout(tmp_path)
    for name in workspace.PROTECTED_DIRS:
        assert (tmp_path / name).is_dir() and not (tmp_path / name).is_symlink()
    ensure_layout(tmp_path)   # idempotent


@pytest.mark.parametrize("name", workspace.PROTECTED_DIRS)
def test_layout_refuses_a_symlinked_protected_dir(tmp_path, name):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / name).symlink_to(outside)
    with pytest.raises(WorkspaceTamperedError):
        ensure_layout(root)
    with pytest.raises(WorkspaceTamperedError):
        verify_layout(root)


def test_verify_layout_refuses_a_missing_or_file_protected_dir(tmp_path):
    ensure_layout(tmp_path)
    (tmp_path / "inputs").rmdir()
    with pytest.raises(WorkspaceTamperedError):
        verify_layout(tmp_path)
    (tmp_path / "inputs").write_text("x")
    with pytest.raises(WorkspaceTamperedError):
        verify_layout(tmp_path)


def test_is_protected(tmp_path):
    ensure_layout(tmp_path)
    root = tmp_path.resolve()
    assert is_protected(root, root / "handles" / "h1.json")
    assert is_protected(root, root / ".tether")
    assert not is_protected(root, root / "outputs" / "a.csv")
    assert not is_protected(root, root)


def test_run_dirs_are_unique_and_under_state(tmp_path):
    ensure_layout(tmp_path)
    a, b = new_run_dir(tmp_path), new_run_dir(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()
    assert a.parent == tmp_path.resolve() / ".tether" / "runs"
    assert scratch_dir(tmp_path) == tmp_path.resolve() / ".tether" / "tmp"
    assert scratch_dir(tmp_path).is_dir()


def test_read_control_file_returns_bytes_or_none(tmp_path):
    assert read_control_file(tmp_path, "emit.json", 100) is None
    (tmp_path / "emit.json").write_bytes(b"{}")
    assert read_control_file(tmp_path, "emit.json", 100) == b"{}"


def test_read_control_file_refuses_a_symlink(tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"k": "v"}')
    run = tmp_path / "run"
    run.mkdir()
    (run / "emit.json").symlink_to(secret)
    with pytest.raises(ControlFileError, match="not a regular file"):
        read_control_file(run, "emit.json", 100)


def test_read_control_file_refuses_directory_and_fifo(tmp_path):
    (tmp_path / "emit.json").mkdir()
    with pytest.raises(ControlFileError, match="not a regular file"):
        read_control_file(tmp_path, "emit.json", 100)
    os.mkfifo(tmp_path / "new_handles.jsonl")
    with pytest.raises(ControlFileError, match="not a regular file"):
        read_control_file(tmp_path, "new_handles.jsonl", 100)   # must not block


def test_read_control_file_enforces_the_cap(tmp_path):
    (tmp_path / "emit.json").write_bytes(b"x" * 101)
    with pytest.raises(ControlFileError, match=r"too large \(101 bytes > 100\)"):
        read_control_file(tmp_path, "emit.json", 100)


def test_read_control_file_refuses_a_name_with_a_separator(tmp_path):
    with pytest.raises(ValueError):
        read_control_file(tmp_path, "../x", 100)


def test_require_regular(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert require_regular(f) == f
    with pytest.raises(ValueError, match="not a regular file"):
        require_regular(tmp_path)
    os.mkfifo(tmp_path / "p")
    with pytest.raises(ValueError, match="not a regular file"):
        require_regular(tmp_path / "p")
    with pytest.raises(FileNotFoundError):
        require_regular(tmp_path / "missing")


def test_iter_bounded_lines_truncates_long_lines_without_miscounting():
    f = io.StringIO("a\n" + "b" * 200_000 + "\nc")
    lines = list(iter_bounded_lines(f, max_line_chars=10))
    assert lines == ["a\n", "b" * 10, "c"]
```

- [ ] **Step 3: Run the tests and see them fail**

Run: `uv run pytest tests/test_workspace.py -p no:warnings`
Expected: collection error, `ModuleNotFoundError: No module named 'tether.workspace'`.

- [ ] **Step 4: Implement `tether/workspace.py`**

```python
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


def verify_layout(root: Path | str) -> None:
    """Raise unless every protected directory is a real directory (never follows a link)."""
    root = Path(root)
    for name in PROTECTED_DIRS:
        if not _is_real_dir(root / name):
            raise WorkspaceTamperedError(
                f"workspace directory {name!r} under {root} is missing or is not a real "
                f"directory; refusing to use this workspace")


def ensure_layout(root: Path | str) -> None:
    """Create the protected directories; refuse a root where one exists as anything else."""
    root = Path(root)
    for name in PROTECTED_DIRS:
        path = root / name
        if not os.path.lexists(path):
            path.mkdir()
    verify_layout(root)


def is_protected(root: Path | str, resolved: Path) -> bool:
    """Whether an already-resolved path lies in (or is) a protected directory of ``root``."""
    try:
        parts = Path(resolved).relative_to(Path(root).resolve()).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in PROTECTED_DIRS


def state_dir(root: Path | str) -> Path:
    return Path(root).resolve() / _STATE


def scratch_dir(root: Path | str) -> Path:
    """Parent-only scratch space (downloads, snapshots). Created on demand."""
    path = state_dir(root) / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_dir(root: Path | str) -> Path:
    """Create and return a fresh per-run exchange directory under ``.tether/runs/``."""
    runs = state_dir(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / secrets.token_hex(8)
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
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ControlFileError("not a regular file")
        if st.st_size > max_bytes:
            raise ControlFileError(f"too large ({st.st_size} bytes > {max_bytes})")
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
        while not piece.endswith("\n"):
            piece = f.readline(chunk)
            if not piece:
                break
            room = max_line_chars - len(kept)
            if room > 0:
                kept += piece[:room]
        yield kept
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_workspace.py -p no:warnings`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tether/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): trust-zone layout and bounded no-follow reads"
```

---
### Task 2: `HandleStore` — thread safety, atomic writes, relocated manifest, `adopt_file`, bounded describers

**Files:**
- Modify: `tether/handles.py`
- Test: `tests/test_handles.py` (append), `tests/test_handles_concurrency.py` (new)

**Interfaces:**
- Consumes: `workspace.ensure_layout`, `workspace.state_dir`, `workspace.WorkspaceTamperedError`
- Produces:
  - `HANDLE_ID_RE: re.Pattern` (`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`)
  - `HandleStore.adopt_file(*, id: str, kind: str, src: Path, source: str) -> Handle` — moves `src` to `handles/<id><ext>`; raises `HandleIdReuseError` for an existing id or destination, `ValueError` for anything else wrong
  - `HandleStore.verify(handle_id: str) -> Path` — digest-checks and returns the handle's absolute path; raises `KeyError`, `HandleTamperedError`
  - `HandleStore._manifest_file` is now `<root>/.tether/manifest.json`
  - All mutating methods are safe to call from multiple threads.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handles_concurrency.py`:

```python
import threading

from tether.handles import HandleStore


def test_concurrent_puts_all_succeed_with_distinct_ids(tmp_path):
    store = HandleStore(tmp_path)
    ids, errors = [], []

    def work():
        for _ in range(50):
            try:
                ids.append(store.put("x" * 100, source="t").id)
            except Exception as e:  # noqa: BLE001 - the test reports any failure
                errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(ids) == 200 and len(set(ids)) == 200
    assert len(HandleStore(tmp_path).manifest()) == 200
```

Append to `tests/test_handles.py`:

```python
# --- trust zones: manifest location, atomic writes, adopt_file, bounded describers ---

def test_manifest_lives_in_state_dir(tmp_path):
    store = HandleStore(tmp_path)
    store.put("hello", source="t")
    assert (tmp_path / ".tether" / "manifest.json").is_file()
    assert not (tmp_path / "handles" / "_manifest.json").exists()


def test_legacy_manifest_is_migrated(tmp_path):
    import json
    (tmp_path / "handles").mkdir()
    (tmp_path / "handles" / "h1.txt").write_text("old data")
    (tmp_path / "handles" / "_manifest.json").write_text(json.dumps(
        {"h1": {"id": "h1", "kind": "text", "path": "handles/h1.txt", "source": "old"}}))
    store = HandleStore(tmp_path)
    assert store.get("h1") == "old data"
    assert (tmp_path / ".tether" / "manifest.json").is_file()
    assert not (tmp_path / "handles" / "_manifest.json").exists()
    assert HandleStore(tmp_path).get("h1") == "old data"      # survives a second reopen


def test_store_refuses_a_symlinked_handles_dir(tmp_path):
    import pytest
    from tether.workspace import WorkspaceTamperedError
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "handles").symlink_to(outside)
    with pytest.raises(WorkspaceTamperedError):
        HandleStore(root)


def test_put_replaces_a_planted_symlink_instead_of_writing_through_it(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    root = tmp_path / "root"
    store = HandleStore(root)
    (root / "handles" / "h1.txt").symlink_to(victim)
    handle = store.put("payload", source="t")
    assert handle.id == "h1"
    assert victim.read_text() == "original"
    assert not (root / "handles" / "h1.txt").is_symlink()
    assert store.get("h1") == "payload"


def test_adopt_file_moves_the_payload_and_derives_metadata(tmp_path):
    store = HandleStore(tmp_path)
    src = tmp_path / "save_1.json"
    src.write_text('{"a": 1}')
    handle = store.adopt_file(id="sales_eu", kind="json", src=src, source="run_python")
    assert handle.path == "handles/sales_eu.json"
    assert not src.exists()
    assert store.get("sales_eu") == {"a": 1}
    assert handle.bytes == 8 and handle.digest


def test_adopt_file_rejects_bad_ids_kinds_and_non_regular_sources(tmp_path):
    import pytest
    store = HandleStore(tmp_path)
    src = tmp_path / "f.txt"
    src.write_text("x")
    for bad in ("", "a/b", "..", ".hidden", "x" * 65, "sp ace", 7):
        with pytest.raises(ValueError):
            store.adopt_file(id=bad, kind="text", src=src, source="s")
    with pytest.raises(ValueError):
        store.adopt_file(id="ok", kind="binary", src=src, source="s")
    link = tmp_path / "link.txt"
    link.symlink_to(src)
    with pytest.raises(ValueError):
        store.adopt_file(id="ok", kind="text", src=link, source="s")
    with pytest.raises(ValueError):
        store.adopt_file(id="ok", kind="text", src=tmp_path / "missing", source="s")
    assert src.read_text() == "x"            # nothing was moved by a rejected call


def test_adopt_file_refuses_reuse_of_an_id_or_destination(tmp_path):
    import pytest
    from tether.handles import HandleIdReuseError
    store = HandleStore(tmp_path)
    first = store.put("one", source="t")                 # h1
    src = tmp_path / "f.txt"
    src.write_text("two")
    with pytest.raises(HandleIdReuseError):
        store.adopt_file(id=first.id, kind="text", src=src, source="s")
    (tmp_path / "handles" / "squat.txt").write_text("already here")
    with pytest.raises(HandleIdReuseError):
        store.adopt_file(id="squat", kind="text", src=src, source="s")
    assert store.get(first.id) == "one"


def test_adopt_file_deletes_a_payload_it_cannot_describe(tmp_path):
    import pytest
    store = HandleStore(tmp_path)
    src = tmp_path / "f.parquet"
    src.write_bytes(b"this is not parquet")
    with pytest.raises(ValueError):
        store.adopt_file(id="bad", kind="dataframe", src=src, source="s")
    assert not (tmp_path / "handles" / "bad.parquet").exists()
    assert "bad" not in store.manifest()


def test_adopt_file_truncates_source(tmp_path):
    store = HandleStore(tmp_path)
    src = tmp_path / "f.txt"
    src.write_text("x")
    handle = store.adopt_file(id="a", kind="text", src=src, source="s" * 5000)
    assert len(handle.source) == 200


def test_dataframe_preview_and_schema_are_capped(tmp_path):
    import pandas as pd
    store = HandleStore(tmp_path)
    wide = pd.DataFrame({f"{i}_{'c' * 300}": ["v" * 5000] for i in range(300)})
    handle = store.put(wide, source="t")
    assert len(handle.preview) <= 900
    assert len(handle.schema) == 200
    assert all(len(name) <= 128 for name in handle.schema)
    assert handle.n_cols == 300


def test_dataframe_preview_reads_only_the_first_batch(tmp_path, monkeypatch):
    import pandas as pd
    import pyarrow.parquet as pq
    store = HandleStore(tmp_path)

    def boom(self, *a, **k):
        raise AssertionError("read_row_group must not be used for previews")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", boom)
    handle = store.put(pd.DataFrame({"a": range(100)}), source="t")
    assert handle.n_rows == 100 and "(5 of 100 rows)" in handle.preview


def test_verify_returns_the_path_and_checks_the_digest(tmp_path):
    import pytest
    from tether.handles import HandleTamperedError
    store = HandleStore(tmp_path)
    handle = store.put("abc", source="t")
    path = store.verify(handle.id)
    assert path == (tmp_path / "handles" / "h1.txt").resolve()
    path.write_text("xyz!")
    with pytest.raises(HandleTamperedError):
        store.verify(handle.id)
    with pytest.raises(KeyError):
        store.verify("nope")
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_handles_concurrency.py tests/test_handles.py -p no:warnings`
Expected: the new tests fail (`FileNotFoundError` in the concurrency test; `AttributeError: 'HandleStore' object has no attribute 'adopt_file'`; manifest path assertions).

- [ ] **Step 3: Implement**

In `tether/handles.py`:

(a) Replace the import block and add constants:

```python
import hashlib
import json
import mimetypes
import os
import re
import stat
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import workspace
from .paths import PathEscapesRootError, safe_path

_PREVIEW_CHARS = 800
_PREVIEW_ROWS = 5
_MAX_SCHEMA_COLS = 200
_MAX_COL_NAME_CHARS = 128
_MAX_SOURCE_CHARS = 200
_DIGEST_BYTES = 64 * 1024
# Uploaded-input previews: raw first lines, for text-like extensions only. Never parsed.
_TEXT_PREVIEW_EXTS = {".csv", ".tsv", ".txt", ".json", ".md"}
_PREVIEW_LINES = 5
_PREVIEW_READ_BYTES = 4096

# Ids that sandboxed code may name a handle with. One path segment, no leading dot, so the
# id can be used directly as a filename stem under handles/.
HANDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_KIND_EXT = {"dataframe": ".parquet", "json": ".json", "text": ".txt"}
```

(b) Replace `HandleStore.__init__`, `_new_id`, `put` and the four `_write_*` methods:

```python
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Creates handles/ and .tether/ as real directories, or refuses the root if either
        # exists as a link: every write below lands inside one of them.
        workspace.ensure_layout(self.root)
        self.dir = self.root / "handles"
        self._state = workspace.state_dir(self.root)
        self._handles: dict[str, Handle] = {}
        self._counter = 0
        # Tool calls in one turn run on worker threads in parallel; every mutation of the
        # id counter, the handle map and the manifest happens under this lock.
        self._lock = threading.RLock()
        self._load_manifest()

    def _new_id(self) -> str:
        self._counter += 1
        return f"h{self._counter}"

    def _atomic(self, dst: Path, write: Callable[[Path], None]) -> None:
        """Write ``dst`` via a unique temp file in the same directory, then rename.

        ``os.replace`` swaps the directory entry, so a link planted at ``dst`` is replaced,
        never written through, and a reader never sees a half-written file.
        """
        fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp_")
        os.close(fd)
        try:
            write(Path(tmp))
            os.chmod(tmp, 0o644)
            os.replace(tmp, dst)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def put(self, obj: Any, source: str, *, id: str | None = None,
            kind: str | None = None, ext: str | None = None) -> Handle:
        kind = kind or self._detect_kind(obj)
        with self._lock:
            hid = id or self._new_id()
            if id is not None:
                self._advance_counter(hid)  # keep auto-ids from colliding with an explicit id
            if kind == "dataframe":
                handle = self._write_dataframe(hid, obj, source)
            elif kind == "json":
                handle = self._write_json(hid, obj, source)
            elif kind == "text":
                handle = self._write_text(hid, obj, source)
            elif kind == "binary":
                handle = self._write_binary(hid, obj, source, ext)
            else:
                raise ValueError(f"unknown handle kind: {kind!r}")
            self._handles[hid] = handle
            self._save_manifest()
            return handle

    def _write_binary(self, hid: str, data: bytes, source: str, ext: str | None) -> Handle:
        # Store raw bytes intact so the file (xls/pdf/image/...) stays readable by pandas,
        # Docling, etc. The extension is preserved so libraries can infer the format.
        rel = f"handles/{hid}{ext or '.bin'}"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_bytes(bytes(data)))
        return Handle(id=hid, kind="binary", path=rel, source=source,
                      **self._describe("binary", path))

    def _write_dataframe(self, hid: str, df: Any, source: str) -> Handle:
        rel = f"handles/{hid}.parquet"
        path = self.root / rel
        self._atomic(path, lambda tmp: df.to_parquet(tmp))
        return Handle(id=hid, kind="dataframe", path=rel, source=source,
                      **self._describe("dataframe", path))

    def _write_json(self, hid: str, obj: Any, source: str) -> Handle:
        # ``default=str`` keeps non-JSON-native types (datetime, Decimal, ...) from
        # crashing serialization, but they round-trip back as strings via get().
        rel = f"handles/{hid}.json"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_text(json.dumps(obj, default=str),
                                                     encoding="utf-8"))
        return Handle(id=hid, kind="json", path=rel, source=source,
                      **self._describe("json", path))

    def _write_text(self, hid: str, obj: str, source: str) -> Handle:
        rel = f"handles/{hid}.txt"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_text(obj, encoding="utf-8"))
        return Handle(id=hid, kind="text", path=rel, source=source,
                      **self._describe("text", path))
```

(c) Replace `_describe_dataframe`:

```python
    def _describe_dataframe(self, path: Path) -> dict[str, Any]:
        """Describe a parquet file without materializing it.

        Schema and row count come from the footer; the preview decodes only the first
        record batch of ``_PREVIEW_ROWS`` rows, never a whole row group (a row group in a
        file sandboxed code wrote can be arbitrarily large). The preview text and the
        schema listing are capped, because cell values and column names are untrusted and
        this summary goes into model context.
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        n_rows = int(pf.metadata.num_rows)
        schema_df = pf.schema_arrow.empty_table().to_pandas()
        batch = next(pf.iter_batches(batch_size=_PREVIEW_ROWS), None)
        shown = 0
        preview = ""
        if batch is not None:
            head = batch.to_pandas()
            shown = len(head)
            preview = head.to_csv(index=False)
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS] + "\n"
        # Caption the rows actually shown, not _PREVIEW_ROWS.
        if n_rows > shown:
            preview += f"... ({shown} of {n_rows} rows)"
        dtypes = list(schema_df.dtypes.items())
        return {
            "bytes": path.stat().st_size,
            "preview": preview,
            "schema": {str(c)[:_MAX_COL_NAME_CHARS]: str(t)
                       for c, t in dtypes[:_MAX_SCHEMA_COLS]},
            "n_rows": n_rows,
            "n_cols": int(len(dtypes)),
        }
```

(d) Wrap the bodies of `register`, `adopt` and `put_input` in `with self._lock:` (indent the existing body one level; change nothing else).

(e) Add `adopt_file` directly after `adopt`:

```python
    def adopt_file(self, *, id: str, kind: str, src: Path, source: str) -> Handle:
        """Move a payload sandboxed code wrote in its run directory into ``handles/`` and
        register it, deriving every described field here.

        The sandbox names only the id, the kind and which file it wrote. The parent picks
        the destination (``handles/<id><ext>``), so no sandbox-supplied path is ever joined
        to the root. ``src`` is ``lstat``-ed and must be a regular file; it is renamed, not
        copied, so a multi-gigabyte save costs nothing. Call only when no sandboxed code is
        running. A payload that cannot be described is deleted and the call raises.
        """
        with self._lock:
            if not isinstance(id, str) or not HANDLE_ID_RE.match(id):
                raise ValueError(f"invalid handle id {id!r}: use letters, digits, '_', '-', "
                                 f"'.', starting with a letter or digit, at most 64 chars")
            ext = _KIND_EXT.get(kind) if isinstance(kind, str) else None
            if ext is None:
                raise ValueError(f"unknown handle kind: {kind!r}")
            rel = f"handles/{id}{ext}"
            dst = self.root / rel
            if id in self._handles or os.path.lexists(dst):
                raise HandleIdReuseError(id)
            try:
                st = os.lstat(src)
            except OSError as e:
                raise ValueError(f"handle payload is missing: {e}") from e
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("handle payload is not a regular file")
            os.rename(src, dst)
            try:
                os.chmod(dst, 0o644)
                described = self._describe(kind, dst)
            except Exception as e:  # noqa: BLE001 - a describer may raise anything on hostile bytes
                dst.unlink(missing_ok=True)
                raise ValueError(f"unreadable {kind} handle payload: {e}") from e
            handle = Handle(id=id, kind=kind, path=rel,
                            source=str(source)[:_MAX_SOURCE_CHARS], **described)
            self._handles[id] = handle
            self._advance_counter(id)
            self._save_manifest()
            return handle
```

(f) Replace `_manifest_file`, `_save_manifest` and `_load_manifest`:

```python
    @property
    def _manifest_file(self) -> Path:
        return self._state / "manifest.json"

    @property
    def _legacy_manifest_file(self) -> Path:
        return self.dir / "_manifest.json"

    def _save_manifest(self) -> None:
        """Persist {id: summary} atomically so a new HandleStore on this root can rehydrate.

        Lives under ``.tether/``, which sandboxed code cannot write in the container tier.
        Rewrites the whole manifest on every mutation -- fine at handle frequency.
        """
        with self._lock:
            payload = json.dumps(self.manifest())
            self._atomic(self._manifest_file,
                         lambda tmp: tmp.write_text(payload, encoding="utf-8"))

    def _read_manifest_records(self, path: Path) -> dict | None:
        """Parse a manifest file. ``None`` if absent, a link, unreadable or malformed."""
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):
                return None
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return records if isinstance(records, dict) else None

    def _load_manifest(self) -> None:
        """Restore handles + the id counter from a prior session on this root.

        Nothing in a manifest is believed: each record is re-derived from its file.
        Tolerant by design -- a corrupt, forged, or orphaned record is skipped, never fatal.
        A manifest left at the pre-trust-zone location (``handles/_manifest.json``) is read
        once, rewritten at the new location and removed.
        """
        records = self._read_manifest_records(self._manifest_file)
        migrating = False
        if records is None:
            records = self._read_manifest_records(self._legacy_manifest_file)
            migrating = records is not None
        if records is None:
            return
        for record in records.values():
            try:
                self._register_record(record)
            except Exception:  # noqa: BLE001 - a describer may raise anything on a corrupt file
                continue
        if migrating:
            self._save_manifest()
            self._legacy_manifest_file.unlink(missing_ok=True)
```

(g) Add `verify` before `get`, and make `get` use it:

```python
    def verify(self, handle_id: str) -> Path:
        """Digest-check a handle and return its absolute path, without loading it."""
        try:
            handle = self._handles[handle_id]
        except KeyError:
            raise KeyError(f"no handle with id {handle_id!r}") from None
        path = safe_path(self.root, handle.path)  # defense-in-depth before any read
        self._verify_digest(handle, path)
        return path

    def get(self, handle_id: str) -> Any:
        path = self.verify(handle_id)
        handle = self._handles[handle_id]
        if handle.kind == "dataframe":
            import pandas as pd
            return pd.read_parquet(path)
        if handle.kind == "json":
            return json.loads(path.read_text(encoding="utf-8"))
        if handle.kind == "binary":
            return str(path)  # binary content is opened by a library; hand back the path
        return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run the handle tests**

Run: `uv run pytest tests/test_handles_concurrency.py tests/test_handles.py -p no:warnings`
Expected: new tests pass. Existing tests that referenced `handles/_manifest.json` (search: `grep -n "_manifest" tests/`) now fail; update each to `tmp_path / ".tether" / "manifest.json"`. A test that forges a manifest must write the forged file at the new location. Do not weaken any assertion.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -p no:warnings`
Expected: only failures that mention the old control-file protocol (fixed in Task 4) or assert on root contents that now include the protected directories. Fix the latter here: a test asserting an empty root or exact root listing must exclude `handles`, `inputs`, `.scripts`, `.tether`.

- [ ] **Step 6: Commit**

```bash
git add tether/handles.py tests/test_handles.py tests/test_handles_concurrency.py tests
git commit -m "feat(handles): thread-safe store, atomic writes, manifest in .tether, adopt_file"
```

---

### Task 3: `tether/procio.py` — bounded process-output capture

**Files:**
- Create: `tether/procio.py`
- Modify: `tether/config.py` (add `SandboxConfig.max_output_bytes`)
- Test: `tests/test_procio.py`

**Interfaces:**
- Produces:
  - `@dataclass class BoundedRun: stdout: str; stderr: str; returncode: int; timed_out: bool`
  - `run_bounded(argv: list[str], *, timeout: float, cap: int, cwd: Path | str | None = None, env: dict[str, str] | None = None, preexec_fn: Callable[[], None] | None = None) -> BoundedRun`
  - `SandboxConfig.max_output_bytes: int = 8192`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_procio.py`:

```python
import sys

from tether.config import SandboxConfig
from tether.procio import run_bounded


def _py(code: str, **kw):
    return run_bounded([sys.executable, "-c", code], **{"timeout": 20, "cap": 1000, **kw})


def test_small_output_is_returned_exactly():
    r = _py("import sys; print('héllo'); print('err', file=sys.stderr)")
    assert r.stdout == "héllo\n" and r.stderr == "err\n"
    assert r.returncode == 0 and not r.timed_out


def test_large_output_keeps_head_and_tail_and_names_the_gap():
    r = _py("import sys; sys.stdout.write('A' * 500 + 'B' * 100000 + 'Z' * 500)")
    assert r.stdout.startswith("A" * 500)
    assert r.stdout.endswith("Z" * 500)
    assert "[tether: 100000 bytes of output omitted]" in r.stdout
    assert len(r.stdout) < 1200


def test_output_exactly_at_the_cap_is_not_marked():
    r = _py("import sys; sys.stdout.write('x' * 1000)")
    assert r.stdout == "x" * 1000


def test_timeout_kills_and_keeps_partial_output():
    r = _py("import sys, time; print('started', flush=True); time.sleep(30)", timeout=1)
    assert r.timed_out and r.returncode != 0
    assert "started" in r.stdout


def test_nonzero_exit_code_is_reported():
    assert _py("raise SystemExit(3)").returncode == 3


def test_config_default():
    assert SandboxConfig().max_output_bytes == 8192
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_procio.py -p no:warnings`
Expected: `ModuleNotFoundError: No module named 'tether.procio'`.

- [ ] **Step 3: Implement**

Add to `SandboxConfig` in `tether/config.py`, after `build_timeout_s`:

```python
    max_output_bytes: int = 8192           # per stream (stdout, stderr): head + tail kept
```

Create `tether/procio.py`:

```python
"""Run a child process with bounded output capture.

``subprocess.run(capture_output=True)`` buffers everything the child prints. The child here
is model-authored code, so each stream is drained through a buffer that keeps only the
first and last halves of a byte cap and counts what it dropped.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 65536
_JOIN_TIMEOUT_S = 5.0


class _HeadTail:
    """Keeps the first ``cap // 2`` and last ``cap - cap // 2`` bytes fed to it."""

    def __init__(self, cap: int) -> None:
        self._head_cap = cap // 2
        self._tail_cap = cap - self._head_cap
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0

    def feed(self, chunk: bytes) -> None:
        self._total += len(chunk)
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                del self._tail[:excess]

    def text(self) -> str:
        dropped = self._total - len(self._head) - len(self._tail)
        if dropped <= 0:
            return bytes(self._head + self._tail).decode("utf-8", errors="replace")
        return (self._head.decode("utf-8", errors="replace")
                + f"\n... [tether: {dropped} bytes of output omitted] ...\n"
                + self._tail.decode("utf-8", errors="replace"))


@dataclass
class BoundedRun:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def _drain(stream, buf: _HeadTail) -> None:
    try:
        while chunk := stream.read1(_CHUNK):
            buf.feed(chunk)
    finally:
        stream.close()


def run_bounded(argv: list[str], *, timeout: float, cap: int,
                cwd: Path | str | None = None, env: dict[str, str] | None = None,
                preexec_fn: Callable[[], None] | None = None) -> BoundedRun:
    """Run ``argv``; return its exit code and at most ``cap`` bytes of each output stream.

    On timeout the process is killed and whatever was captured is returned with
    ``timed_out=True``. The drain threads are daemons and are joined with a bound, so a
    grandchild that keeps a pipe open cannot hang the caller.
    """
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=preexec_fn)
    out, err = _HeadTail(cap), _HeadTail(cap)
    threads = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    return BoundedRun(out.text(), err.text(), proc.returncode, timed_out)
```

- [ ] **Step 4: Run and see them pass**

Run: `uv run pytest tests/test_procio.py tests/test_config.py -p no:warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tether/procio.py tether/config.py tests/test_procio.py
git commit -m "feat(procio): head-and-tail bounded process output capture"
```

---
### Task 4: Run-directory protocol — sandbox orchestration, child helpers, local backend

**Files:**
- Modify: `tether/sandbox.py`, `tether/runtime/tether_sandbox.py`, `tether/runtime/_runner.py`
- Test: `tests/test_sandbox_protocol.py` (new); update `tests/test_sandbox.py`, `tests/test_isolation.py`, `tests/test_deliver.py`, `tests/test_publish.py`, `tests/test_inputs.py` where they encode the old protocol

**Interfaces:**
- Consumes: `workspace.verify_layout`, `workspace.new_run_dir`, `workspace.read_control_file`, `workspace.ControlFileError`, `HandleStore.adopt_file`, `procio.run_bounded`, `paths.validate_segment`
- Produces:
  - `_RunContext(script_rel: str, argv: list[str], root: Path, run_dir: Path, config: SandboxConfig, publish_enabled: bool = False)`
  - `_LaunchResult(stdout: str, stderr: str, exit_code: int, killed_by: str | None = None, orphaned: bool = False)`
  - Child environment: `TETHER_ROOT`, `TETHER_RUN_DIR`, and `TETHER_PUBLISH_ENABLED=1` when publishing is on
  - `ExecResult` is unchanged in shape.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sandbox_protocol.py`:

```python
import json

import pytest

from tether.config import SandboxConfig
from tether.handles import HandleStore
from tether.sandbox import ControlPlaneLimits, LocalSubprocessSandbox
from tether.workspace import WorkspaceTamperedError


def _sandbox(root, **limits):
    store = HandleStore(root)
    sb = LocalSubprocessSandbox(root=root, store=store, config=SandboxConfig(backend="local"),
                                limits=ControlPlaneLimits(**limits) if limits else None)
    return sb, store


def _outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text(json.dumps({"key": "HOST-SECRET"}))
    victim = outside / "victim.txt"
    victim.write_text("original")
    return secret, victim


def test_emit_save_and_load_round_trip(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import pandas as pd\n"
        "from tether_sandbox import emit, save\n"
        "save('df1', pd.DataFrame({'a': [1, 2, 3]}))\n"
        "save('j1', {'k': 'v'})\n"
        "save('t1', 'hello')\n"
        "emit({'ok': True})\n")
    assert res.error is None, res.error
    assert res.result == {"ok": True}
    assert res.new_handles == ["df1", "j1", "t1"]
    assert store.get("j1") == {"k": "v"} and store.get("t1") == "hello"
    assert store.manifest_handles()["df1"].path == "handles/df1.parquet"
    res2 = sb.run_code("from tether_sandbox import load\nlen(load('df1'))\n")
    assert res2.result == 3


def test_run_directory_is_removed_after_the_run(tmp_path):
    root = tmp_path / "root"
    sb, _ = _sandbox(root)
    sb.run_code("from tether_sandbox import emit\nemit(1)\n")
    assert list((root / ".tether" / "runs").iterdir()) == []
    assert not any(p.name.startswith("_") for p in root.iterdir())


def test_emit_symlink_to_a_host_file_is_not_read(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import os\n"
        "p = os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json')\n"
        f"os.symlink({str(secret)!r}, p)\n")
    assert res.result is None
    assert "emit payload not a regular file" in res.error
    assert "HOST-SECRET" not in json.dumps([res.result, res.error, res.stdout])


def test_new_handles_symlink_is_not_read(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import os\n"
        "p = os.path.join(os.environ['TETHER_RUN_DIR'], 'new_handles.jsonl')\n"
        f"os.symlink({str(secret)!r}, p)\n")
    assert res.new_handles == []
    assert "control file not a regular file" in res.error


def test_payload_symlink_is_rejected_and_reported(tmp_path):
    secret, _ = _outside(tmp_path)
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        f"os.symlink({str(secret)!r}, os.path.join(d, 'x.json'))\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'leak', 'kind': 'json', 'file': 'x.json'}) + '\\n')\n")
    assert res.new_handles == []
    assert "1 handle record(s) rejected" in res.error
    assert "leak" not in store.manifest()
    assert secret.exists()                       # the link was not followed or moved


def test_payload_file_name_with_a_separator_is_rejected(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'x', 'kind': 'text', 'file': '../../manifest.json'}) + '\\n')\n")
    assert res.new_handles == [] and "rejected" in res.error
    assert store.manifest() == {}


def test_replacing_handles_dir_with_a_link_is_refused(tmp_path):
    _, victim = _outside(tmp_path)
    root = tmp_path / "root"
    sb, store = _sandbox(root)
    with pytest.raises(WorkspaceTamperedError):
        sb.run_code(
            "import os\n"
            "os.rename('handles', 'handles_real')\n"
            f"os.symlink({str(victim.parent)!r}, 'handles')\n")
    with pytest.raises(WorkspaceTamperedError):
        sb.run_code("1\n")                        # and every later run on this root
    assert victim.read_text() == "original"


def test_save_rejects_invalid_id_in_child(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    res = sb.run_code("from tether_sandbox import save\nsave('eu/sales', {'a': 1})\n")
    assert res.exit_code != 0
    assert "invalid handle id" in res.error and "letters, digits" in res.error
    assert store.manifest() == {}


def test_reused_id_is_reported(tmp_path):
    sb, store = _sandbox(tmp_path / "root")
    store.put("first", source="t", id="dup")
    res = sb.run_code(
        "import json, os\n"
        "d = os.environ['TETHER_RUN_DIR']\n"
        "open(os.path.join(d, 'p.txt'), 'w').write('second')\n"
        "open(os.path.join(d, 'new_handles.jsonl'), 'a').write(\n"
        "    json.dumps({'id': 'dup', 'kind': 'text', 'file': 'p.txt'}) + '\\n')\n")
    assert "handle id already exists" in res.error and "dup" in res.error
    assert store.get("dup") == "first"


def test_emit_over_the_cap_is_rejected(tmp_path):
    sb, _ = _sandbox(tmp_path / "root", max_emit_bytes=100)
    res = sb.run_code("from tether_sandbox import emit\nemit('x' * 500)\n")
    assert res.result is None and "emit payload too large" in res.error


def test_stdout_is_capped(tmp_path):
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code("print('A' * 5_000_000)\n")
    assert len(res.stdout) < 9000
    assert "bytes of output omitted" in res.stdout
    assert len(json.dumps(res.result)) < 9000


def test_timeout_is_reported(tmp_path):
    root = tmp_path / "root"
    store = HandleStore(root)
    sb = LocalSubprocessSandbox(root=root, store=store,
                                config=SandboxConfig(backend="local", timeout_s=1))
    res = sb.run_code("import time\ntime.sleep(30)\n")
    assert res.killed_by == "timeout" and res.exit_code != 0
    assert "tether: killed (timeout)" in res.stderr


def test_publish_requests_travel_through_the_run_dir(tmp_path):
    root = tmp_path / "root"
    sb, _ = _sandbox(root)
    seen = []
    sb.publisher = lambda path, **kw: seen.append((path, kw)) or {"ok": True}
    (root / "outputs").mkdir()
    (root / "outputs" / "a.csv").write_text("x\n")
    res = sb.run_code("from tether_sandbox import publish\npublish('outputs/a.csv', name='A')\n")
    assert res.error is None, res.error
    assert seen == [("outputs/a.csv", {"name": "A", "description": None, "source": "run_python"})]
    assert res.published == [{"ok": True}]


def test_publish_is_unavailable_without_a_publisher(tmp_path):
    sb, _ = _sandbox(tmp_path / "root")
    res = sb.run_code("from tether_sandbox import publish\npublish('x')\n")
    assert res.exit_code != 0 and "publishing is not enabled" in res.error
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_sandbox_protocol.py -p no:warnings`
Expected: failures (`KeyError: 'TETHER_RUN_DIR'` in the child, missing error strings).

- [ ] **Step 3: Rewrite the child helpers**

Replace `tether/runtime/tether_sandbox.py` with:

```python
"""Injected into the sandbox subprocess as the top-level module ``tether_sandbox``.

Communicates with the parent tether only via env vars and a per-run exchange directory:
  TETHER_ROOT             session root directory (handles/ and inputs/ are read-only here)
  TETHER_RUN_DIR          this run's exchange directory; the only place to hand data back
  TETHER_PUBLISH_ENABLED  "1" when the host accepts publish() requests

Files in TETHER_RUN_DIR:
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

_ROOT = Path(os.environ["TETHER_ROOT"])
_RUN = Path(os.environ["TETHER_RUN_DIR"])
_REGISTRY = json.loads((_RUN / "registry.json").read_text(encoding="utf-8"))
_NEW = _RUN / "new_handles.jsonl"
_EMIT = _RUN / "emit.json"
_PUBLISH = _RUN / "publish.jsonl"
_SAVED_THIS_RUN: set[str] = set()  # ids written by save() in this process
# Mirrors tether.handles.HANDLE_ID_RE. Ergonomics only: the parent enforces it for real.
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
    if os.environ.get("TETHER_PUBLISH_ENABLED") != "1":
        raise RuntimeError("publishing is not enabled for this session "
                           "(the host has not configured a publish callback)")
    p = os.path.abspath(os.fspath(path))
    root = os.path.abspath(str(_ROOT))
    if os.path.commonpath([p, root]) == root:
        p = os.path.relpath(p, root)   # convenience: /workspace/x -> x (not a trust step)
    with _PUBLISH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"path": p, "name": name, "description": description}) + "\n")
```

In `tether/runtime/_runner.py` replace `_emit_already_written`:

```python
def _emit_already_written() -> bool:
    path = os.path.join(os.environ.get("TETHER_RUN_DIR", ""), "emit.json")
    return os.path.exists(path) and os.path.getsize(path) > 0
```

- [ ] **Step 4: Rewrite the parent orchestration**

In `tether/sandbox.py`:

(a) Imports: remove `secrets`, `stat`, `subprocess`; add `shutil`. Add:

```python
from . import workspace
from .paths import safe_path, validate_segment
from .procio import run_bounded
from .workspace import ControlFileError
```

Keep `_MAX_PUBLISH_CONTROL_BYTES` and `_MAX_PUBLISH_REQUESTS`.

(b) Replace `_LaunchResult` and `_RunContext`:

```python
@dataclass
class _LaunchResult:
    """What a backend's _launch returns: raw process outcome, pre-parse."""
    stdout: str
    stderr: str
    exit_code: int
    killed_by: str | None = None
    # True when the backend could not confirm the sandbox stopped (a timed-out container it
    # failed to remove). The child may still be writing, so nothing it produced is ingested.
    orphaned: bool = False


@dataclass
class _RunContext:
    """Everything a backend needs to launch one run. Paths are HOST paths; a backend
    translates them to its own namespace."""
    script_rel: str          # user script path relative to root
    argv: list[str]          # user args (already str-coerced)
    root: Path
    run_dir: Path            # per-run exchange directory (.tether/runs/<token>)
    config: SandboxConfig
    publish_enabled: bool = False
```

(c) Replace the `_OrchestratedSandbox` class docstring and the methods `run_code`, `_run_script`, `_ingest_new_handles`, `_process_publications`:

```python
class _OrchestratedSandbox:
    """Shared run orchestration. Subclasses implement ``_launch``.

    Runs are serialized by ``self.lock`` (an ``RLock`` the Session also takes for its own file
    work in the root), so no sandboxed code is running while the parent validates or writes
    paths under the root. Each run gets a fresh exchange directory under ``.tether/runs/``;
    it is the only place the parent reads sandbox-written data from, and every such read is
    no-follow, regular-file-only and size-capped (``workspace.read_control_file``).
    """

    def run_code(self, code: str, args: list[str] | None = None) -> ExecResult:
        with self.lock:
            workspace.verify_layout(self.root)   # .scripts/ must be a real directory
            scripts = self.root / _SCRIPTS_DIR
            # Collision-free across instances/processes; scripts persist as debuggable artifacts.
            fd, abspath = tempfile.mkstemp(prefix="inline_", suffix=".py", dir=scripts)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            os.chmod(abspath, 0o644)
            rel = str(Path(abspath).relative_to(self.root))
            return self.run_script(rel, args)

    def _run_script(self, path: str, args: list[str] | None) -> ExecResult:
        script = safe_path(self.root, path)  # raises PathEscapesRootError if outside
        argv = [str(a) for a in (args or [])]  # coerce so non-str args fail clearly, not opaquely

        workspace.verify_layout(self.root)
        run_dir = workspace.new_run_dir(self.root)
        try:
            registry = {hid: {"kind": h.kind, "path": h.path}
                        for hid, h in self.store.manifest_handles().items()}
            (run_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

            ctx = _RunContext(
                script_rel=str(script.relative_to(self.root)), argv=argv, root=self.root,
                run_dir=run_dir, config=self.config,
                publish_enabled=self.publisher is not None,
            )
            launched = self._launch(ctx)

            if launched.orphaned:
                return ExecResult(
                    stdout=launched.stdout, stderr=launched.stderr, result=None,
                    error=((launched.stderr.strip() + "\n") if launched.stderr.strip() else "")
                          + "tether: the sandbox could not be stopped; nothing it produced "
                            "was ingested",
                    exit_code=launched.exit_code, killed_by=launched.killed_by)

            # The local tier cannot stop a script replacing a protected directory; refuse to
            # ingest anything if one did. (In the container tier they are read-only mounts.)
            workspace.verify_layout(self.root)

            result, emit_error = self._read_emit(run_dir, launched.exit_code)

            # Ergonomic fallback: if the script neither emitted nor ended in an expression but
            # printed something, surface that so a model that just print()s an answer still gets one.
            if (result is None and emit_error is None and launched.exit_code == 0
                    and launched.stdout.strip()):
                result = launched.stdout.strip()

            new_handles, ingest_error = self._ingest_new_handles(run_dir)
            published, publish_error = self._process_publications(run_dir, launched.exit_code)
            base_error = (launched.stderr.strip() or None) if launched.exit_code != 0 else None
            error = "\n".join(p for p in (base_error, emit_error, ingest_error, publish_error)
                               if p) or None

            return ExecResult(stdout=launched.stdout, stderr=launched.stderr, result=result,
                              error=error, exit_code=launched.exit_code,
                              new_handles=new_handles, killed_by=launched.killed_by,
                              published=published)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)   # rmtree never follows a link

    def _read_emit(self, run_dir: Path, exit_code: int) -> tuple[Any | None, str | None]:
        """Parse the child's emit payload; return (result, error). Everything about the file
        is child-controlled, so every failure becomes an error string, never an exception."""
        if exit_code != 0:
            return None, None
        try:
            raw = workspace.read_control_file(run_dir, "emit.json", self.limits.max_emit_bytes)
        except ControlFileError as e:
            return None, f"tether: emit payload {e}; save a handle instead"
        if raw is None:
            return None, None
        try:
            return json.loads(raw.decode("utf-8", errors="replace")), None
        except json.JSONDecodeError as e:
            return None, f"tether: malformed emit payload: {e}"

    def _ingest_new_handles(self, run_dir: Path) -> tuple[list[str], str | None]:
        """Adopt handles the child saved; return (ids, error).

        Records are untrusted: the parent validates the id, picks the destination, moves the
        payload out of the run directory and derives all metadata itself
        (``HandleStore.adopt_file``). Tolerant by design -- a rejected record is counted and
        reported, not fatal. The control file is read only up to ``max_control_bytes`` and at
        most ``max_new_handles`` records are adopted.
        """
        ids: list[str] = []
        try:
            raw = workspace.read_control_file(run_dir, "new_handles.jsonl",
                                              self.limits.max_control_bytes)
        except ControlFileError as e:
            return ids, f"tether: control file {e}; no handles ingested"
        if raw is None:
            return ids, None
        reused: dict[str, None] = {}   # distinct rejected ids, insertion-ordered
        reused_count = 0
        rejected = 0
        cap_error: str | None = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if len(ids) >= self.limits.max_new_handles:
                cap_error = (f"tether: too many new handles (cap "
                             f"{self.limits.max_new_handles}); later records dropped")
                break
            try:
                rec = json.loads(line)
                name = validate_segment(rec["file"], what="handle payload name")
                handle = self.store.adopt_file(id=rec["id"], kind=rec["kind"],
                                               src=run_dir / name,
                                               source=rec.get("source", "run_python"))
                ids.append(handle.id)
            except HandleIdReuseError as e:
                reused_count += 1
                if len(reused) <= _MAX_REPORTED_IDS:  # stop tracking once past the display cap
                    reused[str(e.handle_id)[:_MAX_REPORTED_ID_LEN]] = None
            except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError,
                    OSError):
                rejected += 1
        parts = []
        if reused_count:
            shown = list(reused)[:_MAX_REPORTED_IDS]
            more = ", ..." if len(reused) > _MAX_REPORTED_IDS else ""
            # Bounded by construction: the length cannot grow with the number of records.
            parts.append(f"tether: save rejected for {reused_count} record(s): handle id "
                         f"already exists and handles are immutable; ids: "
                         f"{', '.join(shown)}{more}")
        if rejected:
            parts.append(f"tether: {rejected} handle record(s) rejected (invalid id or kind, "
                         f"or the saved file was missing or unreadable)")
        if cap_error:
            parts.append(cap_error)
        return ids, "\n".join(parts) or None

    def _process_publications(self, run_dir: Path,
                              exit_code: int) -> tuple[list[dict], str | None]:
        """Hand the child's publish requests to the publisher; return (records, error).

        Runs only after the child has exited, and only on a clean exit, so published files are
        complete. Every request is untrusted: the publisher re-validates the path from scratch.
        """
        if self.publisher is None:
            return [], None
        try:
            raw = workspace.read_control_file(run_dir, "publish.jsonl",
                                              _MAX_PUBLISH_CONTROL_BYTES)
        except ControlFileError as e:
            return [], f"tether: publish control file {e}; publication requests ignored"
        if raw is None:
            return [], None
        lines = [ln for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if not lines:
            return [], None
        if exit_code != 0:
            return [], (f"tether: {len(lines)} publication request(s) ignored because the "
                        "script did not exit cleanly")
        notes: list[str] = []
        if len(lines) > _MAX_PUBLISH_REQUESTS:
            notes.append(f"tether: too many publication requests (cap {_MAX_PUBLISH_REQUESTS}); "
                         "later requests dropped")
        records: list[dict] = []
        for line in lines[:_MAX_PUBLISH_REQUESTS]:
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(req, dict):
                continue
            path, name, desc = req.get("path"), req.get("name"), req.get("description")
            if not isinstance(path, str):
                records.append({"error": "invalid publication request (path must be a string)"})
                continue
            try:
                records.append(self.publisher(
                    path, name=name if isinstance(name, str) else None,
                    description=desc if isinstance(desc, str) else None, source="run_python"))
            except Exception as e:  # noqa: BLE001 - one bad request must not lose the run
                records.append({"error": f"publication failed: {type(e).__name__}: {e}"})
        return records, "\n".join(notes) or None
```

Keep `__init__` and `run_script` as they are.

(d) Replace `LocalSubprocessSandbox._launch`:

```python
    def _launch(self, ctx: _RunContext) -> _LaunchResult:
        script_abs = ctx.root / ctx.script_rel
        env = {
            "PATH": _minimal_path(),
            "HOME": str(ctx.root),
            "TMPDIR": str(ctx.root),
            "TETHER_ROOT": str(ctx.root),
            "TETHER_RUN_DIR": str(ctx.run_dir),
            "PYTHONPATH": str(_RUNTIME_DIR),
        }
        if ctx.publish_enabled:
            env["TETHER_PUBLISH_ENABLED"] = "1"
        run = run_bounded([sys.executable, str(_RUNNER), str(script_abs), *ctx.argv],
                          cwd=ctx.root, env=env, timeout=ctx.config.timeout_s,
                          cap=ctx.config.max_output_bytes, preexec_fn=self._limits())
        if run.timed_out:
            return _LaunchResult(run.stdout, run.stderr + "\ntether: killed (timeout)",
                                 -1, killed_by="timeout")
        return _LaunchResult(run.stdout, run.stderr, run.returncode)
```

Delete the now-unused `_as_text` helper only after Task 5 (the container backend still imports it until then).

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest tests/test_sandbox_protocol.py -p no:warnings`
Expected: all pass.

- [ ] **Step 6: Update tests that encode the old protocol**

Run: `uv run pytest -p no:warnings -x -q` repeatedly and fix each failure by these rules (do not delete a test unless its subject no longer exists; do not weaken an assertion):

- `tests/test_sandbox.py::_run_child` and similar helpers that hand-build the child environment: create a run directory and pass the new variables —

```python
    run_dir = root / ".tether" / "runs" / "t"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "registry.json").write_text(json.dumps(registry or {}))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "TETHER_ROOT": str(root),
        "TETHER_RUN_DIR": str(run_dir),
        "PYTHONPATH": str(RUNTIME_DIR),
    }
```

  and read results from `run_dir / "emit.json"` and `run_dir / "new_handles.jsonl"`. Records now carry `file` (a name inside the run directory), not `path`.
- Tests that plant something at `_emit_*`, `_new_handles_*` or `_publish_*` under the root: plant it at the same file name inside `os.environ['TETHER_RUN_DIR']` from the child script instead. Expected error strings: `emit payload not a regular file`, `emit payload too large`, `control file too large`, `control file not a regular file`, `publish control file not a regular file`.
- Tests that had sandboxed code write `handles/<id>.<ext>` and append a `path` record by hand: write the payload into the run directory and append `{"id", "kind", "file"}`.
- Tests asserting a record with an id like `../x` or an absolute path is skipped: keep them; the id now fails `HANDLE_ID_RE` and the error contains `handle record(s) rejected`.
- Tests asserting that stale `_registry_*`/`_emit_*` files are cleaned up: assert instead that `.tether/runs/` is empty after the run.
- Tests that set `TETHER_PUBLISH`: the variable is now `TETHER_PUBLISH_ENABLED=1`.

- [ ] **Step 7: Run the suite (container-argv tests excluded until Task 5)**

Run: `uv run pytest -p no:warnings --deselect tests/test_sandbox_container.py`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tether/sandbox.py tether/runtime tests
git commit -m "feat(sandbox): per-run exchange directory; parent moves payloads into handles/"
```

---
### Task 5: Container backend — read-only protected mounts, `/tether-run`, orphan detection

**Files:**
- Modify: `tether/sandbox_container.py`, `tether/sandbox.py` (delete `_as_text`)
- Test: `tests/test_sandbox_container.py` (update helper + add), `tests/test_container_live.py` (add)

**Interfaces:**
- Consumes: `_RunContext.run_dir`, `_RunContext.publish_enabled`, `_LaunchResult.orphaned`, `workspace.PROTECTED_DIRS`, `procio.run_bounded`
- Produces: container mounts `/workspace` (rw), `/workspace/{handles,inputs,.scripts,.tether}` (ro), `/tether-run` (rw), `/runtime` (ro)

- [ ] **Step 1: Update the argv-test helper and write the failing tests**

In `tests/test_sandbox_container.py` replace `_ctx`:

```python
def _ctx(tmp_path, config, publish_enabled=False):
    return _RunContext(
        script_rel=".scripts/inline_x.py", argv=["EU", "2025"], root=tmp_path,
        run_dir=tmp_path / ".tether" / "runs" / "tok", config=config,
        publish_enabled=publish_enabled,
    )
```

Append:

```python
def _mounts(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def test_run_argv_mounts_protected_dirs_read_only(tmp_path):
    cfg = SandboxConfig(backend="container")
    argv = _sandbox(tmp_path, cfg)._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)
    mounts = _mounts(argv)
    assert f"{tmp_path}:/workspace:rw" in mounts
    for name in ("handles", "inputs", ".scripts", ".tether"):
        assert f"{tmp_path / name}:/workspace/{name}:ro" in mounts
    assert f"{tmp_path / '.tether' / 'runs' / 'tok'}:/tether-run:rw" in mounts
    # the workspace mount comes first so the read-only mounts overlay it
    assert mounts.index(f"{tmp_path}:/workspace:rw") < mounts.index(
        f"{tmp_path / 'handles'}:/workspace/handles:ro")


def test_run_argv_passes_the_run_dir_protocol_env(tmp_path):
    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg), "img:abc", layer=None)
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "TETHER_ROOT=/workspace" in envs and "TETHER_RUN_DIR=/tether-run" in envs
    assert not any(e.startswith(("TETHER_EMIT", "TETHER_REGISTRY", "TETHER_NEW_HANDLES",
                                 "TETHER_PUBLISH")) for e in envs)
    argv = sb._build_run_argv(_ctx(tmp_path, cfg, publish_enabled=True), "img:abc", layer=None)
    assert "TETHER_PUBLISH_ENABLED=1" in [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]


def test_timed_out_container_that_cannot_be_removed_is_orphaned(tmp_path, monkeypatch):
    from tether import sandbox_container
    from tether.procio import BoundedRun

    cfg = SandboxConfig(backend="container")
    sb = _sandbox(tmp_path, cfg)
    monkeypatch.setattr("tether.container_runtime.ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(sandbox_container, "run_bounded",
                        lambda *a, **k: BoundedRun("", "", -9, True))
    monkeypatch.setattr(sb, "_remove_container", lambda name: "\ntether: WARNING: stuck")
    launched = sb._launch(_ctx(tmp_path, cfg))
    assert launched.orphaned and launched.killed_by == "timeout"

    monkeypatch.setattr(sb, "_remove_container", lambda name: "")
    assert not sb._launch(_ctx(tmp_path, cfg)).orphaned
```

Append to `tests/test_container_live.py`:

```python
# --- trust zones: the reproduced escapes, against a real container ---

def _sentinels(croot):
    outside = croot.parent / (croot.name + "-outside")
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text('{"key": "HOST-SECRET"}')
    victim = outside / "victim.txt"
    victim.write_text("original")
    return outside, secret, victim


def test_protected_dirs_are_read_only_and_cannot_be_replaced(croot):
    sb, _ = _container(croot)
    res = sb.run_code(
        "import os\n"
        "out = {}\n"
        "for d in ('handles', 'inputs', '.scripts', '.tether'):\n"
        "    for label, fn in (\n"
        "        ('write', lambda d=d: open(os.path.join(d, 'x'), 'w')),\n"
        "        ('symlink', lambda d=d: os.symlink('/etc/passwd', os.path.join(d, 'l'))),\n"
        "        ('rename', lambda d=d: os.rename(d, d + '_moved')),\n"
        "        ('rmdir', lambda d=d: os.rmdir(d)),\n"
        "    ):\n"
        "        try:\n"
        "            fn(); out[d + ':' + label] = 'ALLOWED'\n"
        "        except OSError as e:\n"
        "            out[d + ':' + label] = 'denied'\n"
        "from tether_sandbox import emit\nemit(out)\n")
    assert res.error is None, res.error
    assert res.result and set(res.result.values()) == {"denied"}, res.result


def test_emit_symlink_escape_is_closed(croot):
    outside, secret, _ = _sentinels(croot)
    try:
        sb, _ = _container(croot)
        res = sb.run_code(
            "import os\n"
            f"os.symlink({str(secret)!r}, os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'))\n")
        assert res.result is None
        assert "HOST-SECRET" not in repr((res.result, res.error, res.stdout))
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_handle_write_escape_is_closed(croot):
    outside, _, victim = _sentinels(croot)
    try:
        sb, store = _container(croot)
        res = sb.run_code(
            "import os\n"
            "errs = []\n"
            "for name in ('handles/h1.txt', '.tether/manifest.json', '.tether/.tmp_x'):\n"
            "    try:\n"
            f"        os.symlink({str(victim)!r}, name)\n"
            "    except OSError as e:\n"
            "        errs.append(name)\n"
            "from tether_sandbox import emit\nemit(errs)\n")
        assert res.result == ["handles/h1.txt", ".tether/manifest.json", ".tether/.tmp_x"]
        handle = store.put("ATTACKER TEXT", source="fetch_url(http://example.invalid)")
        assert handle.id == "h1" and store.get("h1") == "ATTACKER TEXT"
        assert victim.read_text() == "original"
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_save_round_trips_through_the_run_dir(croot):
    sb, store = _container(croot)
    res = sb.run_code(
        "import pandas as pd\n"
        "from tether_sandbox import save\n"
        "save('df1', pd.DataFrame({'a': [1, 2, 3]}))\n")
    assert res.error is None, res.error
    assert res.new_handles == ["df1"] and len(store.get("df1")) == 3
    res = sb.run_code("from tether_sandbox import load\nint(load('df1')['a'].sum())\n")
    assert res.result == 6


def test_direct_write_into_handles_fails_clearly(croot):
    sb, _ = _container(croot)
    res = sb.run_code("open('handles/mine.txt', 'w').write('x')\n")
    assert res.exit_code != 0
    assert "Read-only file system" in res.error or "Permission denied" in res.error
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_sandbox_container.py tests/test_container_live.py -p no:warnings`
Expected: argv tests fail on missing mounts/env; live tests fail (or skip with no runtime — if they skip, start the runtime: these tests are the acceptance test for the whole plan and must be seen passing).

- [ ] **Step 3: Implement**

In `tether/sandbox_container.py`:

(a) Imports: remove `subprocess`' use for the run (keep it for `_remove_container`), drop `_as_text`, add:

```python
from . import workspace
from .procio import run_bounded
```

(b) In `_build_run_argv`, replace everything from the `argv += ["--user", ...` block through the `env` dict and the publish line with:

```python
        argv += [
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ctx.root}:/workspace:rw",
            "-v", f"{_RUNTIME_DIR}:/runtime:ro",
            "-w", "/workspace",
        ]
        # Parent-owned trees are overlaid read-only on the rw workspace. Sandboxed code can
        # then neither write inside them nor replace them (a mount point cannot be renamed
        # or removed from inside the container). The orchestrator has already verified that
        # each is a real directory, so none of these mounts a link's target.
        for name in workspace.PROTECTED_DIRS:
            argv += ["-v", f"{ctx.root / name}:/workspace/{name}:ro"]
        # The one writable channel back to the parent: this run's exchange directory.
        argv += ["-v", f"{ctx.run_dir}:/tether-run:rw"]
        if not cfg.network:
            argv += ["--network", "none"]
        pythonpath = "/runtime"
        if layer is not None:
            argv += ["-v", f"{layer}:/pkgs:ro"]
            pythonpath = "/runtime:/pkgs"
        env = {
            "TETHER_ROOT": "/workspace",
            "TETHER_RUN_DIR": "/tether-run",
            "PYTHONPATH": pythonpath,
            "HOME": "/workspace",
            "TMPDIR": "/tmp",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }
        if ctx.publish_enabled:
            env["TETHER_PUBLISH_ENABLED"] = "1"
```

(the `for key, value in env.items()` loop and the final `argv += [tag, "python", ...]` stay).

(c) Replace `_launch`:

```python
    def _launch(self, ctx: _RunContext) -> _LaunchResult:
        from .container_runtime import ensure_image, ensure_layer, image_tag

        tag = image_tag(self.config.preinstalled)
        ensure_image(self._runtime, tag, self.config)
        layer = ensure_layer(self._runtime, self.config) if self.config.pip_packages else None
        name = f"tether-{secrets.token_hex(8)}"
        argv = self._build_run_argv(ctx, tag, layer, name=name)
        run = run_bounded(argv, timeout=ctx.config.timeout_s,
                          cap=ctx.config.max_output_bytes)
        if run.timed_out:
            # Killing the runtime CLI does not stop the container: remove it by name. If that
            # fails the container may still be running in the bind-mounted root, so the run
            # is reported as orphaned and the orchestrator ingests nothing from it.
            note = self._remove_container(name)
            return _LaunchResult(run.stdout, run.stderr + "\ntether: killed (timeout)" + note,
                                 -1, killed_by="timeout", orphaned=bool(note))
        # 137 == 128 + SIGKILL: the runtime killed the container, typically the memory cap.
        killed_by = "killed" if run.returncode == 137 else None
        return _LaunchResult(run.stdout, run.stderr, run.returncode, killed_by=killed_by)
```

(d) Update the module docstring's mount description to match (b).

(e) In `tether/sandbox.py` delete `_as_text` and any import of it.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_sandbox_container.py tests/test_container_live.py -p no:warnings`
Expected: all pass, with the live tests **running** (not skipped). Existing argv tests that asserted the old `inputs` conditional mount or the `TETHER_EMIT`/`TETHER_REGISTRY`/`TETHER_NEW_HANDLES`/`TETHER_PUBLISH` env values must be updated to the new mounts/env; a test that asserted "a symlinked `inputs/` is not mounted" is replaced by `test_layout_refuses_a_symlinked_protected_dir` (Task 1) plus the `verify_layout` call in `_run_script` — delete it and say so in the commit body.

If `test_protected_dirs_are_read_only_and_cannot_be_replaced` reports any `ALLOWED`, stop: capture the exact `res.result` and the runtime's version (`podman version` / `docker version`) and report back before changing the design. The whole plan rests on this property.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -p no:warnings`
Expected: all pass.

```bash
git add tether/sandbox_container.py tether/sandbox.py tests/test_sandbox_container.py tests/test_container_live.py
git commit -m "feat(container): mount protected dirs read-only; exchange via /tether-run"
```

---

### Task 6: Session and manager — root allocation, id validation, artifacts

**Files:**
- Modify: `tether/session.py`, `tether/manager.py`, `tether/publish.py`
- Test: `tests/test_session.py`, `tests/test_manager.py`, `tests/test_publish.py` (append)

**Interfaces:**
- Consumes: `paths.validate_segment`, `workspace.PROTECTED_DIRS`
- Produces:
  - `session._resolve_root(config: TetherConfig) -> tuple[Path, bool]` — `(root, pre_existing)`; with `root_dir=None` creates `./.tether/runs/run-XXXXXXXX` atomically
  - `Session.artifacts` and `publish.validate_publication` treat `.tether` as internal

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session.py`:

```python
def test_default_roots_are_unique_and_outside_the_conversation_namespace(tmp_path, monkeypatch):
    from tether import Session, TetherConfig
    monkeypatch.chdir(tmp_path)
    a = Session.create(TetherConfig())
    b = Session.create(TetherConfig())
    assert a.root != b.root
    for s in (a, b):
        assert s.root.parent == (tmp_path / ".tether" / "runs").resolve()
        assert s.root.name.startswith("run-")
    assert not (tmp_path / ".tether" / "sessions").exists()


def test_default_root_allocation_is_safe_across_threads(tmp_path, monkeypatch):
    import threading
    from tether import Session, TetherConfig
    monkeypatch.chdir(tmp_path)
    roots = []

    def make():
        roots.append(Session.create(TetherConfig()).root)

    threads = [threading.Thread(target=make) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(roots)) == 8


def test_artifacts_exclude_internal_trees_and_links(tmp_path):
    from tether import Session, TetherConfig
    s = Session.create(TetherConfig(root_dir=tmp_path / "root"))
    s.store.put("x", source="t")
    (s.root / "outputs").mkdir()
    (s.root / "outputs" / "a.csv").write_text("1\n")
    (s.root / "link.txt").symlink_to(s.root / "outputs" / "a.csv")
    assert s.artifacts == ["outputs/a.csv"]
```

Append to `tests/test_manager.py`:

```python
@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "x\x00y", "a\nb", "x" * 300])
def test_conversation_ids_must_be_a_single_safe_segment(tmp_path, bad):
    async def run():
        m = SessionManager(_tether(tmp_path))
        with pytest.raises(ValueError):
            await m.aopen(bad)

    asyncio.run(run())
```

Append to `tests/test_publish.py`:

```python
def test_state_dir_is_not_publishable(tmp_path):
    import pytest
    from tether.publish import PublishError, validate_publication
    (tmp_path / ".tether").mkdir()
    (tmp_path / ".tether" / "manifest.json").write_text("{}")
    with pytest.raises(PublishError, match="internal"):
        validate_publication(tmp_path, ".tether/manifest.json", name=None, description=None,
                             source="t", max_bytes=1000)
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_session.py tests/test_manager.py tests/test_publish.py -p no:warnings`
Expected: the new tests fail (roots under `sessions/N`; the empty-string id case already raises, the control-character and over-long ones do not; `.tether` publishable).

- [ ] **Step 3: Implement**

`tether/session.py` — replace `_resolve_root`:

```python
def _resolve_root(config: TetherConfig) -> tuple[Path, bool]:
    """Return ``(root, pre_existing)``, creating the root.

    A pinned ``root_dir`` is used verbatim. Otherwise a fresh directory is created under
    ``./.tether/runs/`` with ``mkdtemp``, which is atomic: two threads or processes can
    never be handed the same root. That base is deliberately not ``./.tether/sessions/``,
    where conversation roots are named by a caller-supplied id.
    """
    if config.root_dir is not None:
        root = Path(config.root_dir).resolve()
        pre_existing = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        return root, pre_existing
    base = Path.cwd() / ".tether" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base)).resolve(), False
```

In `Session.create` replace the three lines that compute `root`, `pre_existing` and call `root.mkdir(...)` with:

```python
        root, pre_existing = _resolve_root(config)
```

(keep the explanatory comment above it and the `try/except BaseException` cleanup below it unchanged).

Replace the `artifacts` property body:

```python
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(self.root)
            top = rel.parts[0]
            if top in ("handles", ".scripts", ".tether") or top.startswith("_"):
                continue
            out.append(rel.as_posix())
        return out
```

`tether/publish.py` — change the constant:

```python
_INTERNAL_DIRS = ("handles", ".scripts", ".tether")
```

`tether/manager.py` — replace the body of `_conv_root` after the docstring, and import `validate_segment`:

```python
from .paths import safe_path, validate_segment
```

```python
    base = Path(config.root_dir) if config.root_dir else (Path.cwd() / ".tether" / "sessions")
    validate_segment(conv_id, what="conversation id")
    return safe_path(base, conv_id)
```

- [ ] **Step 4: Run, fix encoded expectations, run the suite**

Run: `uv run pytest tests/test_session.py tests/test_manager.py tests/test_publish.py -p no:warnings`
Then update existing tests that assert `".tether/sessions" in str(sess.root)` for a default one-shot root to `".tether/runs"`, and `test_session.py` lines around 146–163 (the "failed start leaves nothing behind" test) to look in `.tether/runs`. Existing tests matching the old message `invalid conversation id (no path separators allowed)` should match `invalid conversation id`.

Run: `uv run pytest -p no:warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tether/session.py tether/manager.py tether/publish.py tests
git commit -m "fix(session): atomic one-shot roots in their own namespace; validate conversation ids"
```

---
### Task 7: File tools — locked, bounded, regular files only, protected-write refusal

**Files:**
- Modify: `tether/tools/files.py`, `tether/tools/search.py`
- Test: `tests/tools/test_files.py`, `tests/tools/test_search.py` (append)

**Interfaces:**
- Consumes: `session.io_lock`, `workspace.require_regular`, `workspace.is_protected`, `workspace.iter_bounded_lines`
- Produces: unchanged tool signatures; `search` clamps `max_matches` to 1000; `list_files` returns at most 2000 paths plus one trailing note string when cut

- [ ] **Step 1: Write the failing tests**

Append to `tests/tools/test_files.py`:

```python
import os
import threading
import time


def test_read_file_rejects_binary(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "blob.bin").write_bytes(b"\x00\x01\x02PK\x03\x04" * 100)
    with pytest.raises(ValueError, match="binary file"):
        read_file(sess, "blob.bin")


def test_read_file_replaces_invalid_utf8_instead_of_raising(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "latin.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    assert read_file(sess, "latin.txt") == "caf�\n"


def test_read_file_refuses_a_fifo_without_blocking(tmp_path):
    sess = _session(tmp_path)
    os.mkfifo(sess.root / "pipe")
    with pytest.raises(ValueError, match="not a regular file"):
        read_file(sess, "pipe")


def test_read_file_does_not_load_a_huge_line_whole(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    (sess.root / "one_line.txt").write_text("x" * 5_000_000)
    from pathlib import Path

    def boom(self, *a, **k):
        raise AssertionError("read_text must not be used")

    monkeypatch.setattr(Path, "read_text", boom)
    out = read_file(sess, "one_line.txt")
    assert len(out) <= 50_000 + 200 and "truncated" in out.lower()


@pytest.mark.parametrize("target", ["handles/h1.txt", "inputs/x/a.csv", ".scripts/s.py",
                                    ".tether/manifest.json"])
def test_write_file_refuses_protected_directories(tmp_path, target):
    sess = _session(tmp_path)
    with pytest.raises(ValueError, match="managed by tether"):
        write_file(sess, target, "x")


def test_write_file_refuses_a_link_into_a_protected_directory(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "alias").symlink_to(sess.root / "handles")
    with pytest.raises(ValueError, match="managed by tether"):
        write_file(sess, "alias/h9.txt", "x")


def test_list_files_skips_state_dir_and_links_and_is_capped(tmp_path):
    sess = _session(tmp_path)
    sess.store.put("x", source="t")
    (sess.root / "a.txt").write_text("1")
    (sess.root / "link").symlink_to(sess.root / "a.txt")
    listed = list_files(sess)
    assert "a.txt" in listed and "handles/h1.txt" in listed
    assert "link" not in listed and not any(p.startswith(".tether") for p in listed)
    (sess.root / "many").mkdir()
    for i in range(2100):
        (sess.root / "many" / f"f{i:04d}").write_text("")
    listed = list_files(sess, "many")
    assert len(listed) == 2001 and "100 more" in listed[-1]


def test_file_tools_wait_for_the_sandbox_lock(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "hi\n")
    done = []
    sess.io_lock.acquire()
    t = threading.Thread(target=lambda: done.append(read_file(sess, "a.txt")))
    t.start()
    time.sleep(0.2)
    assert done == []                      # blocked while a run holds the lock
    sess.io_lock.release()
    t.join(timeout=5)
    assert done == ["hi\n"]
```

Append to `tests/tools/test_search.py` (reuse that file's existing session helper and `search` import; if its helper has a different name, use it): Make sure `import pytest` is at the top of the file.

```python
def test_search_refuses_a_fifo_base(tmp_path):
    import os
    import pytest
    sess = _session(tmp_path)
    os.mkfifo(sess.root / "pipe")
    with pytest.raises(ValueError, match="not a regular file or directory"):
        search(sess, "x", path="pipe")


def test_search_clamps_max_matches(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "big.txt").write_text("hit\n" * 5000)
    assert len(search(sess, "hit", max_matches=10**9)) == 1000


@pytest.mark.skipif(__import__("shutil").which("rg") is None, reason="ripgrep not on PATH")
def test_rg_search_skips_links_fifos_and_oversized_files_and_stops_early(tmp_path, monkeypatch):
    import os
    from tether.tools import search as search_mod
    monkeypatch.setattr(search_mod, "_MAX_SEARCH_FILE_BYTES", 1000)
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("needle outside\n")
    (sess.root / "link.txt").symlink_to(outside)
    os.mkfifo(sess.root / "pipe")
    (sess.root / "big.txt").write_text("needle " + "x" * 2000 + "\n")
    (sess.root / "real.txt").write_text("needle here\n" * 50)
    hits = search(sess, "needle", max_matches=5)
    assert len(hits) == 5 and {h["file"] for h in hits} == {"real.txt"}
    with pytest.raises(ValueError):
        search(sess, "(unclosed")


def test_python_search_skips_links_fifos_and_bounds_lines(tmp_path, monkeypatch):
    import os
    from tether.tools import search as search_mod
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None)   # force the fallback
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("needle outside\n")
    (sess.root / "link.txt").symlink_to(outside)
    os.mkfifo(sess.root / "pipe")
    (sess.root / "real.txt").write_text("x" * 3_000_000 + " needle tail\nneedle here\n")
    hits = search(sess, "needle")
    assert [(h["file"], h["line"]) for h in hits] == [("real.txt", 2)]
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/tools/test_files.py tests/tools/test_search.py -p no:warnings`
Expected: the new tests fail (the FIFO tests may hang without the fix — run with `--timeout` unavailable, so interrupt with Ctrl-C after ~10 s if they block; that hang is the failure).

- [ ] **Step 3: Implement `tether/tools/files.py`**

Replace the whole file:

```python
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
```

- [ ] **Step 4: Implement `tether/tools/search.py`**

Replace everything from `def search(` to the end of the file (keep the module docstring, `_SNIPPET_WIDTH` and `_snippet`), and set the imports to:

```python
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from pathlib import Path

from .. import workspace
from ..paths import safe_path
from ..session import Session

_MAX_MATCHES = 1000
_MAX_SEARCH_FILE_BYTES = 50 * 1024 * 1024
_MAX_LINE_CHARS = 1_000_000
_RG_TIMEOUT_S = 30.0
```

```python
def search(session: Session, pattern: str, path: str = ".", glob: str | None = None,
           ignore_case: bool = False, max_matches: int = 100) -> list[dict]:
    """Search for ``pattern`` (a regex) in files under ``path`` (a file or folder).

    Returns up to ``max_matches`` (at most 1000) hits as ``{"file", "line", "col", "text"}``
    with ``file`` relative to the session root. Uses ripgrep if installed, else a Python
    scan. Links, special files and files over 50 MiB are skipped.
    """
    max_matches = max(1, min(int(max_matches), _MAX_MATCHES))
    with session.io_lock:
        base = safe_path(session.root, path or ".")  # "" / None -> the root
        mode = os.stat(base).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"not a regular file or directory: {path!r}")
        rg = shutil.which("rg")
        if rg:
            return _search_rg(rg, session.root, base, pattern, glob, ignore_case, max_matches)
        return _search_python(session.root, base, pattern, ignore_case, max_matches, glob)


def _search_rg(rg: str, root: Path, base: Path, pattern: str, glob: str | None,
               ignore_case: bool, max_matches: int) -> list[dict]:
    # ripgrep does not follow symlinks and skips non-regular files when walking a directory.
    cmd = [rg, "--json", "--no-heading", "--max-filesize", str(_MAX_SEARCH_FILE_BYTES)]
    if ignore_case:
        cmd.append("-i")
    if glob:
        cmd += ["-g", glob]
    cmd += ["--", pattern, str(base)]
    hits: list[dict] = []
    stopped_early = False
    # stderr goes to a file so a noisy rg can never fill a pipe and deadlock the read loop.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=errf, text=True)
        timer = threading.Timer(_RG_TIMEOUT_S, proc.kill)
        timer.start()
        try:
            for line in proc.stdout:           # streamed: stop reading once we have enough
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") != "match":
                    continue
                data = evt["data"]
                abs_path = Path(data["path"]["text"])
                rel = (abs_path.relative_to(root).as_posix() if abs_path.is_absolute()
                       else abs_path.as_posix())
                # One hit per matching line (first submatch), with a CHARACTER column, to
                # match the Python fallback exactly. rg reports a byte offset, so convert it.
                line_text = data["lines"]["text"]
                sub = data["submatches"][0]
                char_col = len(line_text.encode("utf-8")[:sub["start"]]
                               .decode("utf-8", errors="ignore"))
                hits.append({"file": rel, "line": data["line_number"], "col": char_col,
                             "text": _snippet(line_text, char_col)})
                if len(hits) >= max_matches:
                    stopped_early = True
                    proc.kill()
                    break
        finally:
            timer.cancel()
            proc.stdout.close()
            proc.wait()
        # rg exit codes: 0 = match, 1 = no match, 2 = error (e.g. invalid regex).
        if not stopped_early and proc.returncode == 2:
            errf.seek(0)
            message = errf.read(4096).decode("utf-8", errors="replace").strip()
            if not hits:
                raise ValueError(message or "ripgrep search failed")
        if not stopped_early and proc.returncode < 0:
            raise ValueError(f"search timed out after {_RG_TIMEOUT_S:.0f}s; narrow the path "
                             "or the pattern")
    return hits


def _search_python(root: Path, base: Path, pattern: str, ignore_case: bool,
                   max_matches: int, glob: str | None = None) -> list[dict]:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:  # match the rg path: invalid regex -> ValueError
        raise ValueError(f"invalid search pattern: {e}") from e
    files = [base] if base.is_file() else sorted(base.rglob(glob or "*"))
    hits: list[dict] = []
    for f in files:
        try:
            st = os.lstat(f)               # lstat: a link is skipped, never followed
            if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_SEARCH_FILE_BYTES:
                continue
            with f.open(encoding="utf-8", newline="") as fh:
                for lineno, line in enumerate(
                        workspace.iter_bounded_lines(fh, _MAX_LINE_CHARS), start=1):
                    m = rx.search(line)
                    if m:
                        hits.append({"file": f.relative_to(root).as_posix(), "line": lineno,
                                     "col": m.start(), "text": _snippet(line, m.start())})
                        if len(hits) >= max_matches:
                            return hits
        except (UnicodeDecodeError, OSError):
            continue
    return hits
```

Note for the fallback test: line 1 of `real.txt` is 3,000,012 characters and is cut to 1,000,000, so its "needle tail" is not searched; only line 2 hits. That is the documented bound.

- [ ] **Step 5: Run**

Run: `uv run pytest tests/tools/test_files.py tests/tools/test_search.py -p no:warnings`
Expected: all pass. An existing test that expected `UnicodeDecodeError` from `read_file` on a binary file should now expect `ValueError` matching `binary file`. An existing search test that asserted `ValueError` for an invalid regex must still pass on both paths.

- [ ] **Step 6: Full suite and commit**

Run: `uv run pytest -p no:warnings` — expected: all pass.

```bash
git add tether/tools/files.py tether/tools/search.py tests/tools
git commit -m "fix(tools): file tools hold the session lock, read regular files only, bounded"
```

---

### Task 8: Bounded `run_python` result and `inspect_handle`

**Files:**
- Modify: `tether/tools/code.py`, `tether/tools/inspect.py`, `tether/tools/registry.py` (docstring only)
- Test: `tests/tools/test_code.py`, `tests/tools/test_inspect.py` (append), `tests/test_tool_ceilings.py` (new)

**Interfaces:**
- Consumes: `HandleStore.verify(handle_id) -> Path`, `HandleStore.put`, `TetherConfig.spill_threshold_bytes`
- Produces:
  - `run_python` result dict: when the emitted result's JSON exceeds `spill_threshold_bytes`, `result` becomes `{"spilled": True, "note": str, "handle": <summary dict>}` and the new handle id is appended to `new_handles`
  - `inspect_handle(session, handle_id, rows=20, stats=False) -> dict`: `rows` clamped to 1..100; `preview`/`head` capped at 20 000 characters; `stats=True` adds `"stats": {column: {"min", "max", "null_count"}}` for dataframes

- [ ] **Step 1: Write the failing tests**

Append to `tests/tools/test_code.py` (use that file's session helper; it creates a `Session` on `tmp_path`):

```python
def test_large_emitted_result_is_spilled_to_a_handle(tmp_path):
    import json
    sess = _session(tmp_path)
    out = run_python(sess, code="from tether_sandbox import emit\nemit(list(range(20000)))\n")
    assert out["result"]["spilled"] is True
    hid = out["result"]["handle"]["id"]
    assert hid in out["new_handles"]
    assert sess.store.get(hid) == list(range(20000))
    assert len(json.dumps(out)) < 20_000


def test_small_result_is_returned_inline(tmp_path):
    sess = _session(tmp_path)
    out = run_python(sess, code="6 * 7\n")
    assert out["result"] == 42


def test_printed_output_is_capped_not_spilled(tmp_path):
    import json
    sess = _session(tmp_path)
    out = run_python(sess, code="print('A' * 5_000_000)\n")
    assert len(json.dumps(out)) < 30_000
    assert isinstance(out["result"], str)
```

Append to `tests/tools/test_inspect.py`:

```python
def test_inspect_text_preview_is_capped_for_one_huge_line(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put("B" * 3_000_000, source="s")
    out = inspect_handle(sess, h.id, rows=10**9)
    assert len(out["preview"]) <= 20_000


def test_inspect_rows_are_clamped(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": range(1000)}), source="s")
    assert len(inspect_handle(sess, h.id, rows=10**9)["head"]) == 100
    assert len(inspect_handle(sess, h.id, rows=-5)["head"]) == 1


def test_inspect_head_is_capped_by_size(tmp_path):
    import json
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": ["v" * 5000] * 100}), source="s")
    out = inspect_handle(sess, h.id, rows=100)
    assert len(json.dumps(out["head"])) <= 20_000 and len(out["head"]) >= 1
    assert out["head_truncated"] is True


def test_inspect_never_materializes_the_dataframe(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": [3, 1, None], "s": ["x", "z", "y"]}), source="s")

    def boom(*a, **k):
        raise AssertionError("the parent must not load a dataframe handle")

    monkeypatch.setattr(pd, "read_parquet", boom)
    out = inspect_handle(sess, h.id, rows=2, stats=True)
    assert len(out["head"]) == 2
    assert out["stats"]["a"] == {"min": 1.0, "max": 3.0, "null_count": 1}
    assert out["stats"]["s"] == {"min": "x", "max": "z", "null_count": 0}


def test_inspect_json_preview_is_bounded_raw_text(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put({"k": ["v"] * 100_000}, source="s")
    out = inspect_handle(sess, h.id)
    assert out["preview"].startswith('{"k": ["v", "v"') and len(out["preview"]) <= 20_000


def test_inspect_checks_the_digest(tmp_path):
    from tether.handles import HandleTamperedError
    sess = _session(tmp_path)
    h = sess.store.put("abc", source="s")
    (sess.root / h.path).write_text("tampered!")
    with pytest.raises(HandleTamperedError):
        inspect_handle(sess, h.id)
```

Create `tests/test_tool_ceilings.py`:

```python
"""No built-in tool result may exceed a hard ceiling, whatever the workspace holds."""

import json

import pandas as pd
import pytest

from tether import Session, TetherConfig
from tether.tools.registry import build_tools

_CEILING = 64_000   # bytes of JSON; well under any model context window


@pytest.fixture
def hostile(tmp_path):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"))
    (sess.root / "one_line.txt").write_text("x" * 5_000_000)
    for i in range(3000):
        (sess.root / f"f{i:04d}.txt").write_text("needle " + "y" * 2000 + "\n")
    text = sess.store.put("B" * 3_000_000, source="s")
    frame = sess.store.put(pd.DataFrame({"a": ["v" * 50_000] * 200}), source="s")
    return sess, text.id, frame.id


def _tool(sess, name):
    return next(t for t in build_tools(sess) if t.__name__ == name)


@pytest.mark.parametrize("call", [
    lambda s, t, f: _tool(s, "read_file")("one_line.txt", 0, 10**9),
    lambda s, t, f: _tool(s, "list_files")("."),
    lambda s, t, f: _tool(s, "search")("needle", ".", None, False, 10**9),
    lambda s, t, f: _tool(s, "run_python")("print('A' * 5_000_000)"),
    lambda s, t, f: _tool(s, "run_python")(
        "from tether_sandbox import emit\nemit('Z' * 500_000)"),
    lambda s, t, f: _tool(s, "inspect_handle")(t, 10**9, True),
    lambda s, t, f: _tool(s, "inspect_handle")(f, 10**9, True),
])
def test_builtin_tool_results_stay_under_the_ceiling(hostile, call):
    sess, text_id, frame_id = hostile
    result = call(sess, text_id, frame_id)
    assert len(json.dumps(result, default=str).encode()) <= _CEILING
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/tools/test_code.py tests/tools/test_inspect.py tests/test_tool_ceilings.py -p no:warnings`
Expected: the spill, cap, clamp, stats and ceiling tests fail. (`list_files` and `read_file` rows already pass after Task 7; `search` returns 1000 hits × ≤ 240-char snippets, which exceeds 64 000 — see Step 3c.)

- [ ] **Step 3: Implement**

(a) Replace the body of `tether/tools/code.py` below the imports (add `import json`):

```python
def run_python(session: Session, code: str | None = None, path: str | None = None,
               args: list[str] | None = None) -> dict:
    """Run Python in the sandbox. Provide ``path`` (a script file under the root) or
    ``code`` (inline, written to a scratch script then run). Scripts may use the injected
    ``load(id)`` / ``save(id, obj)`` / ``emit(obj)`` / ``publish(path)`` helpers. Returns the
    ExecResult fields: ``stdout, stderr, result, error, exit_code, new_handles, killed_by,
    published``. ``stdout``/``stderr`` are capped; an emitted ``result`` larger than the
    spill threshold is stored as a handle and replaced by its summary.
    """
    report_progress("running script in sandbox", tool="run_python")
    if path is not None:
        res = session.sandbox.run_script(path, args)
    elif code is not None:
        res = session.sandbox.run_code(code, args)
    else:
        raise ValueError("run_python requires either code= or path=")
    out = asdict(res)
    result = out["result"]
    # The stdout fallback is already capped by the sandbox; only a deliberate emit() (or a
    # final expression) can be large, and that is data, so it becomes a handle.
    from_stdout = isinstance(result, str) and result == out["stdout"].strip()
    if result is not None and not from_stdout:
        encoded = json.dumps(result, default=str)
        if len(encoded.encode()) > session.config.spill_threshold_bytes:
            payload = result if isinstance(result, (dict, list, str)) else encoded
            handle = session.store.put(payload, source="run_python:result")
            out["result"] = {
                "spilled": True,
                "note": "the result was large, so it was stored as a handle; use "
                        "inspect_handle or load() it in run_python",
                "handle": handle.summary(),
            }
            out["new_handles"].append(handle.id)
    return out
```

(b) Replace `tether/tools/inspect.py`:

```python
"""inspect_handle: on-demand deeper look at a stored handle (more preview / stats).

Everything here is bounded and none of it materializes a handle: text and JSON are read
with a capped read, a dataframe contributes only its first record batch and its footer
statistics. Handle bytes may have been written by sandboxed code.
"""

from __future__ import annotations

import json
from typing import Any

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

    No data page is read. A column whose row groups carry no statistics reports ``None``.
    """
    md = pf.metadata
    out: dict[str, dict[str, Any]] = {}
    for j in range(min(md.num_columns, _MAX_STATS_COLS)):
        name = str(md.schema.column(j).path)[:128]
        lo = hi = None
        nulls: int | None = 0
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(j).statistics
            if st is None:
                nulls = None
                continue
            if nulls is not None and st.null_count is not None:
                nulls += st.null_count
            if st.has_min_max:
                try:
                    lo = st.min if lo is None or st.min < lo else lo
                    hi = st.max if hi is None or st.max > hi else hi
                except TypeError:      # incomparable across row groups: report nothing
                    lo = hi = None
        out[name] = {"min": _scalar(lo), "max": _scalar(hi), "null_count": nulls}
    return out
```

(c) In `tether/tools/search.py` bound the total result size (1000 hits of 240-character snippets would be ~300 KB): add a size budget and apply it in `search()` —

```python
_MAX_SEARCH_RESULT_CHARS = 60_000


def _fit(hits: list[dict]) -> list[dict]:
    """Keep the longest prefix of ``hits`` whose JSON stays under the result budget."""
    size = 0
    for i, hit in enumerate(hits):
        size += len(json.dumps(hit)) + 2
        if size > _MAX_SEARCH_RESULT_CHARS:
            return hits[:i] + [{"note": f"{len(hits) - i} more matches omitted; narrow the "
                                        "pattern or the path"}]
    return hits
```

  and return `_fit(...)` around both `_search_rg(...)` and `_search_python(...)` calls in `search()`. Add this test to `tests/tools/test_search.py` first and see it fail:

```python
def test_search_result_is_capped_by_total_size(tmp_path):
    import json
    sess = _session(tmp_path)
    (sess.root / "big.txt").write_text(("needle " + "y" * 2000 + "\n") * 1000)
    hits = search(sess, "needle", max_matches=1000)
    assert len(json.dumps(hits)) <= 61_000
    assert "more matches omitted" in hits[-1]["note"]
```

  `test_search_clamps_max_matches` from Task 7 uses 4-character lines (1000 hits ≈ 57 KB of JSON, under the 60 000 budget) and still returns exactly 1000.

(d) In `tether/tools/registry.py` update the `inspect_handle` wrapper docstring:

```python
        """Deeper look at a handle: up to `rows` (max 100) preview lines or head rows; for a
        dataframe, stats=True adds per-column min/max/null_count."""
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/tools tests/test_tool_ceilings.py -p no:warnings`
Expected: all pass. The existing `test_inspect_dataframe_with_stats` asserted a `describe` key: change it to assert `out["stats"]` has an entry per column with `min`, `max`, `null_count`.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -p no:warnings` — expected: all pass.

```bash
git add tether/tools tests
git commit -m "fix(tools): cap run_python and inspect_handle output; no dataframe load in the parent"
```

---
### Task 9: `publish_handle` — convert dataframes inside the sandbox

**Files:**
- Modify: `tether/tools/deliver.py`
- Test: `tests/test_deliver.py` (append + update)

**Interfaces:**
- Consumes: `session.sandbox.run_code(code: str, args: list[str]) -> ExecResult`, `session.io_lock`, `session.publish(...)`, `HandleStore.verify`
- Produces: unchanged tool signature and result shape; the parent no longer loads a dataframe handle to convert it

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deliver.py`:

```python
def test_publish_handle_never_loads_the_dataframe_in_the_parent(tmp_path, monkeypatch):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), source="t")

    def boom(*a, **k):
        raise AssertionError("the parent must not load a dataframe handle")

    monkeypatch.setattr(pd, "read_parquet", boom)
    from tether.tools.deliver import publish_handle
    out = publish_handle(sess, h.id, "csv", "report")
    assert "error" not in out, out
    assert host.copies["report.csv"] == b"a,b\n1,x\n2,y\n"
    assert not [p for p in (sess.root / "outputs").iterdir() if p.name.startswith(".publish_")]


def test_publish_handle_reports_a_failed_conversion(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    (sess.root / h.path).write_bytes(b"not parquet any more")     # digest mismatch
    from tether.tools.deliver import publish_handle
    out = publish_handle(sess, h.id, "csv")
    assert "error" in out and host.calls == []
```

Replace the existing `test_publish_handle_xlsx_without_openpyxl_is_an_error` with:

```python
def test_publish_handle_xlsx_reports_a_missing_engine(tmp_path):
    import importlib.util
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    from tether.tools.deliver import publish_handle
    out = publish_handle(sess, h.id, "xlsx")
    if importlib.util.find_spec("openpyxl") is None:
        assert "openpyxl" in out["error"] and host.calls == []
    else:
        assert "error" not in out and host.calls[0].name.endswith(".xlsx")
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_deliver.py -p no:warnings -k "never_loads or failed_conversion or missing_engine"`
Expected: `AssertionError: the parent must not load a dataframe handle`.

- [ ] **Step 3: Implement**

In `tether/tools/deliver.py` set the imports to:

```python
import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path

from ..handles import HandleTamperedError
from ..paths import PathEscapesRootError, safe_filename, safe_path
from ..session import Session
```

Add below `_DATAFRAME_FORMATS`:

```python
# Runs inside the sandbox. Fixed text: the handle id, format and output path arrive as
# argv, never by string interpolation, so nothing model-supplied is ever executed as code.
_CONVERT_SCRIPT = """\
import sys
from tether_sandbox import load

handle_id, fmt, out = sys.argv[1:4]
df = load(handle_id)
if fmt == "csv":
    df.to_csv(out, index=False)
elif fmt == "xlsx":
    df.to_excel(out, index=False)
elif fmt == "parquet":
    df.to_parquet(out)
else:
    df.to_json(out, orient="records")
"""
```

Replace `_publish_handle`'s dataframe branch and delete `_write_dataframe`:

```python
def _publish_handle(session: Session, handle_id: str, format: str | None, name: str | None,
                    description: str | None) -> dict:
    handle = session.store.manifest_handles().get(handle_id)
    if handle is None:
        return {"error": f"no handle with id {handle_id!r}"}
    try:
        if handle.kind == "dataframe":
            fmt = (format or "csv").lower()
            if fmt not in _DATAFRAME_FORMATS:
                return {"error": f"unsupported format {format!r}; use one of {_DATAFRAME_FORMATS}"}
            filename = safe_filename(name, fallback=f"{handle_id}.{fmt}")
            if not filename.lower().endswith(f".{fmt}"):
                filename += f".{fmt}"
            session.store.verify(handle_id)        # digest check; the bytes are not loaded here
            rel = _convert_in_sandbox(session, handle_id, fmt, filename)
        else:
            backing = session.store.verify(handle_id)
            ext = backing.suffix.lstrip(".").lower()
            if format is not None and format.lower() != ext:
                return {"error": f"a {handle.kind} handle is delivered as-is (.{ext}); "
                                 "format conversion applies to dataframe handles only"}
            filename = safe_filename(name, fallback=backing.name)
            rel = _write_output(session, filename, lambda p: shutil.copyfile(backing, p))
    except (OSError, ValueError, ImportError, HandleTamperedError) as e:
        return {"error": f"could not write {handle_id} for publishing: {e}"}
    return session.publish(rel, name=filename, description=description,
                           source="tool:publish_handle")


def _outputs_dir(session: Session) -> Path:
    try:
        outputs = safe_path(session.root, _OUTPUTS)
    except PathEscapesRootError as e:
        raise ValueError("outputs/ resolves outside the workspace") from e
    outputs.mkdir(exist_ok=True)
    return outputs


def _convert_in_sandbox(session: Session, handle_id: str, fmt: str, filename: str) -> str:
    """Convert a dataframe handle to ``outputs/<filename>`` by running a fixed script in the
    sandbox, so the parent never parses or materializes sandbox-authored parquet.

    The script writes to a name the parent just invented (so nothing can be pre-planted
    there); once it has exited, the parent checks the result is a regular file and renames
    it over the destination, which replaces a planted link instead of writing through it.
    The caller holds the session lock.
    """
    outputs = _outputs_dir(session)
    tmp_name = f".publish_{secrets.token_hex(8)}.{fmt}"
    tmp = outputs / tmp_name
    try:
        res = session.sandbox.run_code(_CONVERT_SCRIPT,
                                       [handle_id, fmt, f"{_OUTPUTS}/{tmp_name}"])
        if res.exit_code != 0:
            lines = [ln for ln in (res.error or "").splitlines() if ln.strip()]
            raise ValueError(lines[-1] if lines else f"conversion exited {res.exit_code}")
        if not stat.S_ISREG(os.lstat(tmp).st_mode):
            raise ValueError("conversion did not produce a regular file")
        os.replace(tmp, outputs / filename)
    finally:
        if os.path.lexists(tmp) and not os.path.isdir(tmp):
            tmp.unlink()
    return f"{_OUTPUTS}/{filename}"
```

Change `_write_output` to use `_outputs_dir`:

```python
def _write_output(session: Session, filename: str, write) -> str:
    """Write ``outputs/<filename>`` without ever following a link sandboxed code planted."""
    outputs = _outputs_dir(session)
    fd, tmp = tempfile.mkstemp(dir=outputs, prefix=".publish_")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, outputs / filename)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return f"{_OUTPUTS}/{filename}"
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_deliver.py -p no:warnings`
Expected: all pass, including the pre-existing `test_publish_handle_does_not_write_through_planted_symlink` and `test_publish_handle_refuses_symlinked_outputs_dir` unchanged.

- [ ] **Step 5: Commit**

```bash
git add tether/tools/deliver.py tests/test_deliver.py
git commit -m "fix(deliver): convert dataframe handles inside the sandbox, not in the parent"
```

---

### Task 10: Streaming egress — `guarded_fetch`, deadline, byte ceiling

**Files:**
- Modify: `tether/egress.py`, `tether/config.py`, `tether/tools/fetch.py`, `tether/tools/documents.py`
- Test: `tests/test_egress.py`, `tests/tools/test_fetch.py`, `tests/test_documents.py`

**Interfaces:**
- Consumes: `workspace.scratch_dir`, `workspace.require_regular`, `session.io_lock`
- Produces:
  - `FetchConfig.total_timeout_s: float = 120.0`, `FetchConfig.max_bytes_ceiling: int = 100_000_000`
  - `class FetchDeadlineExceeded(httpx.HTTPError)`
  - `@dataclass class FetchedBody: status_code: int; url: str; headers: httpx.Headers; encoding: str; body: bytes; truncated: bool` with property `is_error -> bool`
  - `guarded_fetch(url: str, cfg: FetchConfig, *, client: httpx.Client, max_bytes: int, resolve: Callable[[str], list[str]] | None = None, clock: Callable[[], float] = time.monotonic) -> FetchedBody`
  - `guarded_get` is removed.

- [ ] **Step 1: Write the failing tests**

In `tests/test_egress.py` change the import to `from tether.egress import (BlockedAddressError, FetchDeadlineExceeded, guarded_fetch, validate_url)` and mechanically convert every `guarded_get(` call to `guarded_fetch(` with an added `max_bytes=10_000` keyword; where a test read `resp.text` use `resp.body.decode()`, and `resp.status_code` is unchanged. Rename the tests `test_guarded_get_*` → `test_guarded_fetch_*`. Then append:

```python
def test_guarded_fetch_stops_reading_at_max_bytes():
    served = []

    def stream():
        for _ in range(1000):
            served.append(1)
            yield b"x" * 1000

    def handler(request):
        return httpx.Response(200, content=stream())

    with _client(handler) as c:
        got = guarded_fetch("https://example.com/big", FetchConfig(), client=c,
                            max_bytes=2500, resolve=_public())
    assert got.truncated and len(got.body) == 2500
    assert len(served) < 1000                       # the rest was never pulled


def test_guarded_fetch_enforces_a_total_deadline():
    ticks = iter(range(0, 10_000, 50))              # each clock() call advances 50 s

    def handler(request):
        return httpx.Response(200, content=(b"x" for _ in range(100)))

    with _client(handler) as c:
        with pytest.raises(FetchDeadlineExceeded):
            guarded_fetch("https://example.com/slow", FetchConfig(total_timeout_s=120),
                          client=c, max_bytes=10_000, resolve=_public(),
                          clock=lambda: next(ticks))


def test_guarded_fetch_rejects_a_redirect_to_a_disallowed_scheme():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://example.org/plain"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError, match="scheme"):
            guarded_fetch("https://example.com/start", FetchConfig(allowed_schemes=("https",)),
                          client=c, max_bytes=10_000, resolve=_public())


def test_guarded_fetch_reports_encoding_and_final_url():
    def handler(request):
        return httpx.Response(200, content="é".encode("latin-1"),
                              headers={"content-type": "text/plain; charset=latin-1"})

    with _client(handler) as c:
        got = guarded_fetch("https://example.com/x", FetchConfig(), client=c,
                            max_bytes=100, resolve=_public())
    assert got.encoding == "latin-1" and got.url == "https://example.com/x"
    assert got.body.decode(got.encoding) == "é" and not got.truncated and not got.is_error
```

(`_client` and `_public` are that file's existing helpers for a mock-transport client and a public-address resolver; use whatever names the file already has.)

Append to `tests/tools/test_fetch.py`:

```python
def test_fetch_url_clamps_model_supplied_max_bytes(tmp_path):
    sess = _session(tmp_path, max_bytes_ceiling=5000)

    def handler(request):
        return httpx.Response(200, content=b"a" * 20_000,
                              headers={"content-type": "text/plain"})

    summary = fetch_url(sess, "https://example.com/big", max_bytes=10**12,
                        client=_client(handler))
    assert summary["truncated"] is True and summary["bytes"] == 5000


def test_fetch_url_does_not_buffer_past_the_limit(tmp_path):
    sess = _session(tmp_path, max_bytes=3000)
    pulled = []

    def stream():
        for _ in range(500):
            pulled.append(1)
            yield b"z" * 1000

    def handler(request):
        return httpx.Response(200, content=stream(), headers={"content-type": "text/plain"})

    summary = fetch_url(sess, "https://example.com/big", client=_client(handler))
    assert summary["truncated"] is True and len(pulled) < 500


def test_fetch_url_reports_full_size_from_content_length(tmp_path):
    sess = _session(tmp_path, max_bytes=10)

    def handler(request):
        return httpx.Response(200, content=b"a" * 100, headers={"content-type": "text/plain"})

    summary = fetch_url(sess, "https://example.com/x", client=_client(handler))
    assert summary["truncated"] is True and summary["full_bytes"] == 100


def test_fetch_url_binary_over_limit_is_an_error(tmp_path):
    sess = _session(tmp_path, max_bytes=10)

    def handler(request):
        return httpx.Response(200, content=b"\x89PNG\x00\xff" * 100,
                              headers={"content-type": "image/png"})

    out = fetch_url(sess, "https://example.com/i.png", client=_client(handler))
    assert "exceeds max_bytes" in out["error"]


def test_fetch_url_deadline_is_a_structured_error(tmp_path, monkeypatch):
    from tether.egress import FetchDeadlineExceeded
    sess = _session(tmp_path)

    def late(*a, **k):
        raise FetchDeadlineExceeded("fetch exceeded the 120s total deadline")

    monkeypatch.setattr("tether.tools.fetch.guarded_fetch", late)
    out = fetch_url(sess, "https://example.com/x")
    assert "total deadline" in out["error"]
```

Append to `tests/test_documents.py` (use the file's existing session helper):

```python
def test_read_document_refuses_a_truncated_download(tmp_path, monkeypatch):
    import httpx
    from tether.egress import FetchedBody
    from tether.tools.documents import read_document
    sess = _session(tmp_path)
    monkeypatch.setattr(
        "tether.tools.documents.guarded_fetch",
        lambda url, cfg, **k: FetchedBody(200, url, httpx.Headers({}), "utf-8", b"%PDF", True))
    out = read_document(sess, "https://example.com/big.pdf",
                        convert=lambda p: (_ for _ in ()).throw(AssertionError("no convert")))
    assert "larger than" in out["error"]


def test_read_document_downloads_into_private_scratch_and_cleans_up(tmp_path, monkeypatch):
    import httpx
    from tether.egress import FetchedBody
    from tether.tools.documents import read_document
    sess = _session(tmp_path)
    monkeypatch.setattr(
        "tether.tools.documents.guarded_fetch",
        lambda url, cfg, **k: FetchedBody(200, url, httpx.Headers({}), "utf-8", b"%PDF-1", False))
    seen = []

    def convert(path):
        seen.append(path)
        return "# doc"

    out = read_document(sess, "https://example.com/a.pdf", convert=convert)
    assert out["kind"] == "text"
    assert "/.tether/tmp/" in seen[0] and seen[0].endswith(".pdf")
    assert list((sess.root / ".tether" / "tmp").iterdir()) == []
    assert not (sess.root / "_documents").exists()


def test_read_document_converts_a_private_snapshot_of_a_workspace_file(tmp_path):
    from tether.tools.documents import read_document
    sess = _session(tmp_path)
    (sess.root / "report.pdf").write_bytes(b"%PDF-local")
    seen = []

    def convert(path):
        seen.append((path, open(path, "rb").read()))
        return "# local"

    out = read_document(sess, "report.pdf", convert=convert)
    assert out["kind"] == "text"
    assert "/.tether/tmp/" in seen[0][0] and seen[0][0].endswith(".pdf")
    assert seen[0][1] == b"%PDF-local"
    assert list((sess.root / ".tether" / "tmp").iterdir()) == []


def test_read_document_refuses_a_fifo(tmp_path):
    import os
    from tether.tools.documents import read_document
    sess = _session(tmp_path)
    os.mkfifo(sess.root / "pipe.pdf")
    out = read_document(sess, "pipe.pdf", convert=lambda p: "x")
    assert "not a regular file" in out["error"]
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_egress.py tests/tools/test_fetch.py tests/test_documents.py -p no:warnings`
Expected: `ImportError: cannot import name 'guarded_fetch'`.

- [ ] **Step 3: Implement the config and egress**

`tether/config.py`, in `FetchConfig`:

```python
    max_redirects: int = 5
    total_timeout_s: float = 120.0        # wall-clock bound on one fetch, all hops included
    max_bytes_ceiling: int = 100_000_000  # hard cap on a per-call max_bytes the model supplies
```

`tether/egress.py`: add `import time`, `from dataclasses import dataclass`; replace `guarded_get` (and update the module docstring's mention of it) with:

```python
class FetchDeadlineExceeded(httpx.HTTPError):
    """A fetch ran past ``FetchConfig.total_timeout_s`` (all redirect hops included)."""


@dataclass
class FetchedBody:
    """A fetched response whose body was read with a byte cap."""
    status_code: int
    url: str                 # final URL, after redirects
    headers: httpx.Headers
    encoding: str            # charset from the headers, else utf-8
    body: bytes              # at most max_bytes
    truncated: bool          # the server had more than max_bytes to send

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400


def guarded_fetch(url: str, cfg: FetchConfig, *, client: httpx.Client, max_bytes: int,
                  resolve: Callable[[str], list[str]] | None = None,
                  clock: Callable[[], float] = time.monotonic) -> FetchedBody:
    """GET ``url``, validating the initial address and every redirect hop, and read at most
    ``max_bytes`` of the body.

    Redirects are followed here rather than by httpx, because httpx would follow them
    without consulting egress policy; ``follow_redirects=False`` is forced per request.
    Every hop's scheme must be in ``cfg.allowed_schemes``. The body is streamed: reading
    stops at ``max_bytes`` (the response is then closed, so the remainder is never pulled),
    and the whole operation is bounded by ``cfg.total_timeout_s`` -- the client's own
    timeout only bounds each read, which a slow drip never trips. The caller owns the
    client's lifecycle.
    """
    deadline = clock() + cfg.total_timeout_s

    def check_deadline() -> None:
        if clock() > deadline:
            raise FetchDeadlineExceeded(
                f"fetch exceeded the {cfg.total_timeout_s:.0f}s total deadline: {url!r}")

    current = url
    for _ in range(cfg.max_redirects + 1):
        check_deadline()
        scheme = urlparse(current).scheme
        if scheme not in cfg.allowed_schemes:
            raise BlockedAddressError(
                f"scheme {scheme!r} not allowed for {current!r} (allowed: {cfg.allowed_schemes})")
        validate_url(current, cfg, resolve=resolve)
        with client.stream("GET", current, follow_redirects=False) as resp:
            location = resp.headers.get("location")
            if resp.status_code in _REDIRECT_CODES and location:
                current = urljoin(current, location)  # relative -> absolute, then re-validate
                continue
            body = bytearray()
            truncated = False
            for chunk in resp.iter_bytes():
                body += chunk
                if len(body) > max_bytes:
                    del body[max_bytes:]
                    truncated = True
                    break
                check_deadline()
            return FetchedBody(resp.status_code, str(resp.url), resp.headers,
                               resp.encoding or "utf-8", bytes(body), truncated)
    raise httpx.HTTPError(f"too many redirects (> {cfg.max_redirects}) starting at {url!r}")
```

- [ ] **Step 4: Implement `fetch_url`**

In `tether/tools/fetch.py`: `import json`; import `guarded_fetch` instead of `guarded_get`; replace the function body from `cfg = session.config.fetch` to the end:

```python
    cfg = session.config.fetch
    # The per-call limit is model-supplied; the ceiling is the host's.
    limit = cfg.max_bytes if max_bytes is None else max(1, min(int(max_bytes),
                                                               cfg.max_bytes_ceiling))

    scheme = urlparse(url).scheme
    if scheme not in cfg.allowed_schemes:
        # Structured, like every other failure here and like read_document's handling of
        # exactly this case.
        return {"error": f"scheme {scheme!r} not allowed (allowed: {cfg.allowed_schemes})",
                "status": None, "url": url}

    report_progress(f"fetching {url}", tool="fetch_url")

    owns_client = client is None
    client = client or _default_client(cfg)
    try:
        try:
            got = guarded_fetch(url, cfg, client=client, max_bytes=limit)
        except BlockedAddressError as e:
            return {"error": f"blocked by egress policy: {e}", "status": None, "url": url}
        except httpx.HTTPError as e:
            return {"error": f"request failed: {e}", "status": None, "url": url}
    finally:
        if owns_client:
            client.close()

    if got.is_error:  # 4xx/5xx -> structured error the model can adapt to
        return {"error": f"HTTP {got.status_code} for {got.url}",
                "status": got.status_code, "url": got.url}

    body = got.body
    content_type = got.headers.get("content-type", "")

    # Binary payloads (xls/pdf/image/zip/...) are stored as raw bytes so they stay readable
    # by pandas/Docling; a truncated binary file is useless, so that is an error.
    if _is_binary(content_type, body):
        if got.truncated:
            return {"error": f"binary response exceeds max_bytes ({limit}); call again with "
                             f"a larger max_bytes (ceiling {cfg.max_bytes_ceiling})",
                    "url": url}
        handle = session.store.put(body, source=f"fetch_url({url})", kind="binary",
                                   ext=_ext_for(url, content_type))
        return handle.summary()

    text = body.decode(got.encoding, errors="replace")
    handle = None
    if "json" in content_type and not got.truncated:
        try:
            handle = session.store.put(json.loads(text), source=f"fetch_url({url})",
                                       kind="json")
        except ValueError:
            handle = None                 # mislabelled or malformed: keep it as text
    if handle is None and "html" in content_type and not raw:
        import trafilatura
        md = trafilatura.extract(text, output_format="markdown")
        handle = session.store.put(md if md else text, source=f"fetch_url({url})", kind="text")
    if handle is None:
        handle = session.store.put(text, source=f"fetch_url({url})", kind="text")

    summary = handle.summary()
    if got.truncated:
        summary["truncated"] = True
        declared = got.headers.get("content-length", "")
        if declared.isdigit():
            summary["full_bytes"] = int(declared)
    return summary
```

`store.put(json.loads(text), kind="json")` requires a dict or list; a JSON scalar (e.g. `42`) would be stored by `_write_json` as-is, which is fine.

Catch `LookupError` too when decoding with a server-named charset Python does not know: wrap the decode —

```python
    try:
        text = body.decode(got.encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
```

  and add this test to `tests/tools/test_fetch.py` first:

```python
def test_fetch_url_survives_an_unknown_charset(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, content=b"plain",
                              headers={"content-type": "text/plain; charset=x-nonsense"})

    summary = fetch_url(sess, "https://example.com/x", client=_client(handler))
    assert sess.store.get(summary["id"]) == "plain"
```

- [ ] **Step 5: Implement `read_document`**

In `tether/tools/documents.py`: import `guarded_fetch` instead of `guarded_get`, add `from .. import workspace`; replace `_download` and the two source branches of `read_document`:

```python
def _scratch_file(session: Session, suffix: str) -> Path:
    """A fresh file in the parent-only scratch directory (never in the model's work area)."""
    fd, abspath = tempfile.mkstemp(prefix="doc_", suffix=suffix,
                                   dir=workspace.scratch_dir(session.root))
    os.close(fd)
    return Path(abspath)


def _download(session: Session, url: str) -> Path:
    """Fetch ``url`` through the egress guard into private scratch; return the local path.

    The filename keeps the URL's extension so Docling can infer the format. A download
    that hits the byte cap is refused: converting a partial document yields garbage.
    """
    cfg = session.config.fetch
    client = httpx.Client(timeout=cfg.timeout_s, follow_redirects=False,
                          headers={"User-Agent": _USER_AGENT})
    try:
        got = guarded_fetch(url, cfg, client=client, max_bytes=cfg.max_bytes)
    finally:
        client.close()
    if got.is_error:  # 4xx/5xx -> reported as a download failure by the caller
        raise httpx.HTTPError(f"HTTP {got.status_code}")
    if got.truncated:
        raise httpx.HTTPError(f"document is larger than FetchConfig.max_bytes ({cfg.max_bytes})")
    dest = _scratch_file(session, _safe_suffix(url))
    dest.write_bytes(got.body)
    return dest


def _snapshot(session: Session, source: str) -> Path:
    """Copy a workspace file into private scratch under the session lock.

    The model's work area is writable by sandboxed code, so the path is checked and the
    bytes are copied while no sandboxed code can run; conversion (slow) then happens on
    the private copy, outside the lock.
    """
    limit = session.config.max_input_bytes
    with session.io_lock:
        target = workspace.require_regular(safe_path(session.root, source))
        if target.stat().st_size > limit:
            raise ValueError(f"document too large ({target.stat().st_size} bytes > {limit})")
        suffix = target.suffix if _SAFE_SUFFIX.match(target.suffix) else ".bin"
        dest = _scratch_file(session, suffix)
        try:
            with target.open("rb") as src, dest.open("wb") as out:
                copied = 0
                while chunk := src.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > limit:
                        raise ValueError(f"document too large (more than {limit} bytes)")
                    out.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
    return dest
```

In `read_document`, rename `downloaded` to `scratch` throughout and replace the source handling:

```python
    cfg = session.config.fetch
    scratch: Path | None = None
    scheme = urlparse(source).scheme
    if scheme in cfg.allowed_schemes:
        # Docling fetches URLs itself and follows its own redirects, which egress policy
        # cannot see. Download through the guard instead and convert the local file.
        try:
            scratch = _download(session, source)
        except BlockedAddressError as e:
            return {"error": f"blocked by egress policy: {e}", "source": source}
        except httpx.HTTPError as e:
            return {"error": f"could not download {source!r}: {e}", "source": source}
        except OSError as e:
            return {"error": f"could not store {source!r}: {e}", "source": source}
    elif scheme == "":
        try:
            scratch = _snapshot(session, source)
        except PathEscapesRootError:
            return {"error": f"path escapes the workspace root: {source!r}", "source": source}
        except (OSError, ValueError) as e:
            return {"error": f"could not read {source!r}: {e}", "source": source}
    else:
        return {"error": f"unsupported source scheme {scheme!r}; pass a workspace path or a "
                         f"URL with one of {cfg.allowed_schemes}", "source": source}
    target = str(scratch)
```

and the `finally` block becomes unconditional cleanup of the scratch file:

```python
    finally:
        # Scratch only: the markdown lives in a handle, and leaving these behind would give
        # a model looping on read_document an unbounded disk-growth primitive.
        scratch.unlink(missing_ok=True)
```

`PathEscapesRootError` is a `ValueError` subclass: keep its `except` clause first, as written.

- [ ] **Step 6: Run and update fakes**

Run: `uv run pytest tests/test_egress.py tests/tools/test_fetch.py tests/test_documents.py -p no:warnings`
Existing `tests/test_documents.py` fakes patch `tether.tools.documents.guarded_get` and return an `httpx.Response`: patch `tether.tools.documents.guarded_fetch` instead, with signature `(url, cfg, *, client, max_bytes, resolve=None)`, returning `FetchedBody(status, url, httpx.Headers({}), "utf-8", body_bytes, False)`. Tests that asserted the converter receives the workspace path itself now assert it receives a path under `.tether/tmp/` with the same suffix and bytes. Tests asserting downloads land in `_documents/` assert `.tether/tmp/`.
Expected after updates: all pass.

- [ ] **Step 7: Full suite and commit**

Run: `uv run pytest -p no:warnings` — expected: all pass.

```bash
git add tether/egress.py tether/config.py tether/tools/fetch.py tether/tools/documents.py tests
git commit -m "fix(egress): stream fetches with a byte cap, a total deadline and per-hop scheme checks"
```

---
### Task 11: Conversations — AG-UI turns under the lock, activity-aware TTL

**Files:**
- Modify: `tether/conversation.py`, `tether/manager.py`, `tether/api.py`
- Test: `tests/test_conversation.py`, `tests/test_manager.py` (append)

**Interfaces:**
- Consumes: `tether.agui.agui_event_stream(agent, bus, input_data, **kwargs) -> AsyncIterator`
- Produces:
  - `Conversation.busy -> bool` (property)
  - `Conversation.astream_agui(input_data: dict, **agui_kwargs) -> AsyncIterator[Any]`
  - `SessionManager` never expires a busy conversation; `aopen` of a live id refreshes `last_activity`
  - `Tether.agui_stream` delegates to `Conversation.astream_agui`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_conversation.py`:

```python
def _conv(tmp_path):
    import asyncio
    from tether import TetherConfig
    from tether.conversation import Conversation
    from tether.testing import StubChatClient, text

    async def make():
        return await Conversation.acreate(
            id="c1", config=TetherConfig(root_dir=tmp_path / "c1"),
            client=StubChatClient([text("ok")]), bundles=("files",))
    return make


def _fake_stream(events, gate=None):
    async def fake(agent, bus, input_data, **kwargs):
        for e in events:
            if gate is not None:
                await gate.wait()
            yield e
    return fake


def test_agui_turn_holds_the_lock_and_stamps_activity(tmp_path, monkeypatch):
    import asyncio

    async def run():
        conv = await _conv(tmp_path)()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["a", "b"]))
        conv.last_activity = 0.0
        seen, busy = [], []
        async for event in conv.astream_agui({"messages": []}):
            seen.append(event)
            busy.append(conv.busy)
        assert seen == ["a", "b"] and busy == [True, True]
        assert not conv.busy and conv.last_activity > 0.0
        await conv.aclose()

    asyncio.run(run())


def test_agui_turns_on_one_conversation_are_serialized(tmp_path, monkeypatch):
    import asyncio

    async def run():
        conv = await _conv(tmp_path)()
        gate = asyncio.Event()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["x"], gate))
        order = []

        async def turn(label):
            async for _ in conv.astream_agui({"messages": []}):
                order.append(label)

        first = asyncio.create_task(turn("first"))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(turn("second"))
        await asyncio.sleep(0.05)
        assert conv.busy and order == []
        gate.set()
        await asyncio.gather(first, second)
        assert order == ["first", "second"]
        await conv.aclose()

    asyncio.run(run())


def test_abandoned_agui_stream_releases_the_lock(tmp_path, monkeypatch):
    import asyncio

    async def run():
        conv = await _conv(tmp_path)()
        monkeypatch.setattr("tether.agui.agui_event_stream", _fake_stream(["a", "b", "c"]))
        stream = conv.astream_agui({"messages": []})
        assert await stream.__anext__() == "a"
        assert conv.busy
        await stream.aclose()                     # what a server does on client disconnect
        assert not conv.busy
        assert (await conv.aask("still usable?")).error is None
        await conv.aclose()

    asyncio.run(run())
```

Append to `tests/test_manager.py`:

```python
def test_busy_conversation_is_never_expired(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=0.01)
        conv = await m.aopen("t1")
        async with conv._lock:                    # a turn is in flight
            await asyncio.sleep(0.05)             # well past the TTL
            await m.sweep()
            assert m.get("t1") is conv and conv.session.root.exists()
            assert await m.aopen("t1") is conv
        await m.aclose()

    asyncio.run(run())


def test_reopening_a_live_conversation_counts_as_activity(tmp_path):
    async def run():
        m = SessionManager(_tether(tmp_path), idle_ttl_s=60)
        conv = await m.aopen("t1")
        conv.last_activity -= 30
        before = conv.last_activity
        assert await m.aopen("t1") is conv
        assert conv.last_activity > before
        await m.aclose()

    asyncio.run(run())


def test_agui_stream_keeps_an_active_thread_alive_past_its_ttl(tmp_path, monkeypatch):
    async def run():
        h = Tether(TetherConfig(root_dir=tmp_path / "base", idle_ttl_s=0.5),
                   client=StubChatClient([text("x")]))

        async def fake(agent, bus, input_data, **kwargs):
            yield "event"

        monkeypatch.setattr("tether.agui.agui_event_stream", fake)
        convs = []
        for turn in range(4):                      # 4 turns, 0.2 s apart: 0.6 s > TTL overall
            async for _ in h.agui_stream({"threadId": "t1", "messages": []}):
                pass
            conv = h._sessions().get("t1")
            convs.append(conv)
            if turn == 0:
                (conv.session.root / "kept.txt").write_text("from turn 0")
            await asyncio.sleep(0.2)
        assert all(c is convs[0] for c in convs)   # never reaped and recreated
        assert (convs[0].session.root / "kept.txt").read_text() == "from turn 0"
        await h.aclose_sessions()

    asyncio.run(run())
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_conversation.py tests/test_manager.py -p no:warnings`
Expected: `AttributeError: 'Conversation' object has no attribute 'astream_agui'`; the busy and TTL tests fail on reaping.

- [ ] **Step 3: Implement**

`tether/conversation.py` — add `AsyncIterator` to the typing import (`from typing import TYPE_CHECKING, Any, AsyncIterator`) and add after `aask`:

```python
    @property
    def busy(self) -> bool:
        """True while a turn (``aask`` or an AG-UI stream) holds this conversation."""
        return self._lock.locked()

    async def astream_agui(self, input_data: dict, **agui_kwargs: Any) -> AsyncIterator[Any]:
        """Run one AG-UI turn: serialized with every other turn on this conversation, and
        counted as activity at its start and its end so an idle TTL measures real idleness.

        The lock is released when the stream is exhausted **or closed**: a server that
        closes this generator on client disconnect frees the conversation for the next turn.
        """
        from . import agui

        async with self._lock:
            self.last_activity = time.monotonic()
            try:
                async for event in agui.agui_event_stream(
                        self.agent, self.session.status_bus, input_data, **agui_kwargs):
                    yield event
            finally:
                self.last_activity = time.monotonic()
```

`tether/manager.py` — replace `_expired` and the reuse branch of `aopen`:

```python
    def _expired(self, conv: Conversation) -> bool:
        # A conversation with a turn in flight is by definition not idle.
        if self._ttl is None or conv.busy:
            return False
        return (time.monotonic() - conv.last_activity) > self._ttl
```

```python
                if existing is not None and not self._expired(existing):
                    # Opening is activity: without this, a caller could be handed a
                    # conversation that a sweep reaps before its turn takes the lock.
                    existing.last_activity = time.monotonic()
                    return existing
```

`tether/api.py` — replace the tail of `agui_stream` (from the `# No single-flight lock here` comment to the end of the method), and remove the now-unused `from .agui import agui_event_stream` line at the top of the method:

```python
        conv = await self._sessions().aopen(thread_id, tools=tools)
        # The conversation serializes turns and stamps activity; see Conversation.astream_agui.
        async for event in conv.astream_agui(input_data, **agui_kwargs):
            yield event
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_conversation.py tests/test_manager.py tests/test_agui.py -p no:warnings`
Expected: all pass. If an existing test patches `tether.api.agui_event_stream` or `tether.agui.agui_event_stream` by a different route, point it at `tether.agui.agui_event_stream`.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -p no:warnings` — expected: all pass.

```bash
git add tether/conversation.py tether/manager.py tether/api.py tests
git commit -m "fix(conversation): AG-UI turns take the conversation lock and count as activity"
```

---

### Task 12: Side-effect-free constructor; image built for the host's Python

**Files:**
- Modify: `tether/api.py`, `pyproject.toml`, `tether/runtime/Containerfile`, `tether/container_runtime.py`
- Test: `tests/test_api.py`, `tests/test_container_runtime.py` (append)

**Interfaces:**
- Produces: `Tether(config).config` is a copy when a search key is resolved; `os.environ` is untouched by construction; image builds pass `--build-arg PYTHON_VERSION=<major.minor>`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_constructor_reads_dotenv_without_exporting_it(tmp_path, monkeypatch):
    import os
    from tether import Tether, TetherConfig
    from tether.testing import StubChatClient, text
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SOME_OTHER_SECRET", raising=False)
    (tmp_path / ".env").write_text("TAVILY_API_KEY=tvly-from-file\nSOME_OTHER_SECRET=s3cret\n")
    cfg = TetherConfig()
    h = Tether(cfg, client=StubChatClient([text("x")]))
    assert h.config.search.api_key == "tvly-from-file"
    assert cfg.search.api_key is None                  # the caller's object is untouched
    assert "TAVILY_API_KEY" not in os.environ and "SOME_OTHER_SECRET" not in os.environ


def test_environment_key_wins_over_dotenv_and_explicit_key_wins_over_both(tmp_path, monkeypatch):
    from tether import Tether, TetherConfig
    from tether.config import SearchConfig
    from tether.testing import StubChatClient, text
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TAVILY_API_KEY=from-file\n")
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    client = StubChatClient([text("x")])
    assert Tether(TetherConfig(), client=client).config.search.api_key == "from-env"
    explicit = TetherConfig(search=SearchConfig(api_key="explicit"))
    assert Tether(explicit, client=client).config is explicit
```

Append to `tests/test_container_runtime.py`:

```python
def test_image_build_targets_the_host_python_version():
    import sys
    import types
    from tether.config import SandboxConfig
    from tether.container_runtime import ensure_image

    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        missing = argv[1:3] == ["image", "inspect"]
        return types.SimpleNamespace(returncode=1 if missing else 0, stdout="", stderr="")

    ensure_image("podman", "tether-sandbox:x",
                 SandboxConfig(backend="container", build_on_demand=True), run)
    build = calls[-1]
    assert build[:2] == ["podman", "build"]
    want = f"PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}"
    assert want in build and build[build.index(want) - 1] == "--build-arg"


def test_containerfile_takes_the_python_version_as_an_argument():
    from tether.container_runtime import _CONTAINERFILE
    text = _CONTAINERFILE.read_text()
    assert "ARG PYTHON_VERSION=3.12" in text
    assert "FROM python:${PYTHON_VERSION}-slim" in text
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_api.py tests/test_container_runtime.py -p no:warnings`
Expected: the four new tests fail.

- [ ] **Step 3: Implement**

`tether/api.py` — add `import dataclasses` and `import os` at the top; replace the tail of `Tether.__init__` (from `self.config = config or TetherConfig()` — keep the other attribute assignments) so that the config handling reads:

```python
        cfg = config or TetherConfig()
        if cfg.search.api_key is None:
            key = os.environ.get("TAVILY_API_KEY") or _dotenv_value("TAVILY_API_KEY")
            if key:
                # A copy: constructing a Tether must not rewrite the caller's config, and
                # must not export a .env file into this process's environment.
                cfg = dataclasses.replace(
                    cfg, search=dataclasses.replace(cfg.search, api_key=key))
        self.config = cfg
```

and delete the old `if self.config.search.api_key is None:` block with its `load_dotenv()` call. Add at module level, above `class Tether`:

```python
def _dotenv_value(name: str) -> str | None:
    """Read one value from ./.env without touching os.environ."""
    from dotenv import dotenv_values

    return dotenv_values(".env").get(name)
```

`pyproject.toml` — add to `dependencies`, after `"httpx>=0.27",`:

```toml
    "python-dotenv>=1.0",   # read TAVILY_API_KEY from ./.env without exporting the file
```

then run `uv sync --prerelease=allow` (a `uv.lock` change limited to the new direct-dependency edge is expected).

`tether/runtime/Containerfile` — replace the whole file:

```dockerfile
# Base sandbox image: Python + the preinstalled libraries.
# The tether runtime/ helpers are NOT baked in -- they are mounted read-only at /runtime,
# so the image never drifts from the installed tether version.
# PYTHON_VERSION is set by the builder to the host interpreter's major.minor, so the image
# tag and the pip-layer cache key (both keyed on the host version) describe the interpreter
# that actually runs sandboxed code.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
ARG PREINSTALLED="pandas pyarrow numpy httpx"
RUN pip install --no-cache-dir ${PREINSTALLED}
```

`tether/container_runtime.py` — in `ensure_image`, replace the `build = [...]` list:

```python
    build = [runtime, "build", "-t", tag,
             "--build-arg", f"PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
             "--build-arg", f"PREINSTALLED={' '.join(config.preinstalled)}",
             "-f", str(_CONTAINERFILE), str(_RUNTIME_DIR)]
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_api.py tests/test_container_runtime.py tests/test_search_config.py -p no:warnings`
Expected: all pass. An existing test asserting the exact build argv list must include the new `--build-arg` pair. An existing test asserting `h.config is cfg` when a key came from the environment must compare field values instead.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -p no:warnings` — expected: all pass.

```bash
git add tether/api.py pyproject.toml uv.lock tether/runtime/Containerfile tether/container_runtime.py tests
git commit -m "fix: constructor no longer exports .env or mutates config; image built for host python"
```

---

### Task 13: CI — lint and tests on every push, and before publishing

**Files:**
- Create: `.github/workflows/tests.yml`
- Modify: `.github/workflows/publish.yml`, `tests/test_paths.py`, `tests/test_sandbox.py`, `tests/tools/test_web.py`

- [ ] **Step 1: Make lint clean**

Run: `uv run ruff check tether tests`
Expected before: the pre-existing findings — `F401` unused `pathlib.Path` in `tests/test_paths.py`, and `E402` late imports in `tests/test_sandbox.py` (around line 81–86) and `tests/tools/test_web.py` (around line 78). (`tether/tools/files.py` was fixed by the Task 7 rewrite.)

Fix: run `uv run ruff check tether tests --fix` for the `F401`; for each `E402`, move the flagged `import`/`from` lines up into the import block at the top of that file (merge with existing imports; do not add `# noqa`).

Run: `uv run ruff check tether tests`
Expected: `All checks passed!`

- [ ] **Step 2: Add the workflow**

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
  pull_request:
  workflow_call:

jobs:
  test:
    name: Lint and test (Python ${{ matrix.python }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python }}

      # The search tool prefers ripgrep; install it so that code path is tested, not only
      # the Python fallback.
      - name: Install ripgrep
        run: sudo apt-get update && sudo apt-get install -y ripgrep

      - name: Sync
        run: uv sync --prerelease=allow

      - name: Lint
        run: uv run ruff check tether tests

      # The suite pins the sandbox to the local tier (tests/conftest.py) and needs no
      # network or keys. Container-tier tests skip themselves when no runtime is usable.
      - name: Test
        run: uv run pytest -p no:warnings
```

- [ ] **Step 3: Gate publishing on the tests**

In `.github/workflows/publish.yml`, add a job before `build` and make `build` need it:

```yaml
jobs:
  test:
    name: Lint and test
    uses: ./.github/workflows/tests.yml

  build:
    name: Build distributions
    needs: test
    runs-on: ubuntu-latest
```

(the rest of `build` and `publish` is unchanged).

- [ ] **Step 4: Verify locally**

Run: `uv run ruff check tether tests && uv run pytest -p no:warnings`
Expected: lint clean; all tests pass.
Run: `uv run python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ('.github/workflows/tests.yml','.github/workflows/publish.yml')]; print('ok')"`
Expected: `ok` (if `yaml` is not importable, run `uvx --from pyyaml python -c ...` with the same body).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows tests/test_paths.py tests/test_sandbox.py tests/tools/test_web.py
git commit -m "ci: run lint and tests on push and pull request; gate publishing on them"
```

---
### Task 14: Documentation — README, CHANGELOG, ROADMAP

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md`

No code. Each step is an exact text change; after all of them, Step 7 verifies.

- [ ] **Step 1: README — security section**

In `README.md` under `### Security and confinement`, insert this bullet immediately after the "Layer 2 — Executed code" bullet:

```markdown
- **Workspace trust zones.** The session root is bind-mounted into the sandbox, so the harness never trusts what it finds directly under it. Four directories are owned by the harness and mounted **read-only** into the container: `handles/` (handle files), `inputs/` (uploads), `.scripts/` (inline scripts) and `.tether/` (the manifest, scratch files and per-run exchange directories). Sandboxed code can neither write inside them nor replace them, and `write_file` refuses paths that resolve into them. Each run gets one writable exchange directory (`/tether-run`): `save()`, `emit()` and `publish()` write there, and the harness reads it back by fixed file name with no-follow, regular-file-only, size-capped opens, then **moves** saved payloads into `handles/` itself. No path supplied by sandboxed code is ever joined to the root. Before and after every run the harness checks that the four directories are still real directories and raises `WorkspaceTamperedError` otherwise.
```

Replace the entire "Handle integrity" bullet with:

```markdown
- **Handle integrity.** Metadata the model sees for a handle (schema, row count, preview) is derived by the harness from the bytes on disk **when the handle is created**, never reported by the sandboxed code that wrote them — and it is re-derived, not believed, when a manifest is rehydrated onto a root that outlived its process. The handle **record** is immutable: reusing an existing id is refused and reported. In the container tier the handle **bytes** are immutable too, because `handles/` is a read-only mount. As defence in depth (and as the only tamper signal in the `local` tier), a content digest is recorded when the metadata is derived and checked on every read; a mismatch raises `HandleTamperedError`. The digest (sha256, version-tagged) covers the file's length, its first 64 KiB and its last 64 KiB, so it is not a whole-file check. Handle ids chosen by sandboxed code must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. The parent↔sandbox exchange is bounded (1 MiB emit payload, 8 MiB control file, 256 new handles per run; see the configuration table).
- **Bounded results and reads.** Every built-in tool result has a ceiling: `run_python` keeps the head and tail of `stdout`/`stderr` (`SandboxConfig.max_output_bytes` each) and stores a large emitted result as a handle; `inspect_handle`, `read_file`, `list_files` and `search` cap what they return. The harness reads sandbox-written and fetched data with bounded memory: files are streamed, only regular files are opened (a FIFO cannot block it), a dataframe handle is never materialized in the host process (previews read one record batch, `inspect_handle(stats=True)` reads footer statistics, and `publish_handle` converts inside the sandbox), and HTTP bodies are streamed up to the byte limit under a total deadline.
- **Residual risks.** The harness still parses the parquet footer and first record batch of a file sandboxed code wrote, and an oversized data page is decoded whole. Container-tier disk use is unbounded (`max_file_size_mb` applies to the `local` tier only). DNS rebinding between validation and connect is not closed. The `local` tier provides none of the guarantees in this section.
```

In the "Layer 3 — Egress" bullet, replace the sentence beginning "The harness follows redirects itself" with:

```markdown
The harness follows redirects itself (`follow_redirects=False` is forced on every request), so **every hop is re-validated**, including its scheme against `FetchConfig.allowed_schemes`. Bodies are streamed: reading stops at the byte limit, a model-supplied `max_bytes` is clamped to `FetchConfig.max_bytes_ceiling`, and the whole fetch is bounded by `FetchConfig.total_timeout_s`.
```

- [ ] **Step 2: README — sandbox tiers, sessions, configuration, layout**

Under `### Sandbox tiers`, in the `container` bullet replace "The session root is bind-mounted to `/workspace`, so handles round-trip exactly as in the local tier." with:

```markdown
The session root is bind-mounted to `/workspace` with `handles/`, `inputs/`, `.scripts/` and `.tether/` overlaid read-only, plus one writable per-run exchange directory at `/tether-run`; a script that writes into `handles/` directly gets a read-only error — use `save(id, obj)`.
```

In the "Notes:" paragraph replace "(the default `~/.tether/...` is)" with "(the default root is under the current directory, so run from somewhere inside your home directory, or set `root_dir`)".

In the configuration table replace the `root_dir` row and add rows:

```markdown
| `root_dir` | `None` | `None` → a fresh one-shot dir `./.tether/runs/run-XXXXXXXX`; continuous conversations use `./.tether/sessions/<id>` (or `<root_dir>/<id>`) |
| `sandbox.max_output_bytes` | `8192` | Bytes of `stdout` and of `stderr` kept from a `run_python` call (first and last halves; the gap is reported) |
| `fetch.total_timeout_s` | `120` | Wall-clock bound on one fetch, all redirect hops included |
| `fetch.max_bytes_ceiling` | `100 MB` | Upper bound on the per-call `max_bytes` the model may pass to `fetch_url` |
```

Under `### API keys`, after the `.env` code block add:

```markdown
`TAVILY_API_KEY` is read from the environment, then from `./.env`. Constructing a `Tether` does not export `.env` into the process environment.
```

Under `### AG-UI and CopilotKit`, after the paragraph that begins "Point your AG-UI client at this endpoint", add:

```markdown
Each request is one conversation turn: turns on the same `threadId` are serialized, and a turn counts as activity for `idle_ttl_s`. Close the event stream on client disconnect (Starlette's `StreamingResponse` does) so the conversation is released for the next turn. The `threadId` selects the workspace, so authenticate and authorize it in your endpoint before calling `agui_stream`.
```

In the tool-surface table replace the `inspect_handle` row:

```markdown
| `inspect_handle(id, rows=20, stats=False)` | Deeper on-demand look at a handle: up to 100 preview lines / head rows; `stats=True` adds per-column min/max/null_count for dataframes |
```

In `## Project layout` add these lines in the listing (after `paths.py`):

```
  workspace.py   trust zones: protected dirs, per-run exchange dir, bounded no-follow reads
  procio.py      child-process runner with bounded output capture
```

In `## Development` add after the code block:

```markdown
CI (`.github/workflows/tests.yml`) runs lint and the suite on every push and pull request, and publishing requires it.
```

- [ ] **Step 3: CHANGELOG**

In `CHANGELOG.md`, under `## Unreleased`, fix the stale digest sentence — replace "The digest covers the file's length plus its first
  64 KiB." with "The digest covers the file's length, its first 64 KiB and its last 64 KiB." — and insert this section as the first subsection under `## Unreleased`:

```markdown
### Workspace trust zones and bounded I/O
- **Security fix:** sandboxed code could plant a symlink in the session root that the
  harness then read or wrote through, reaching host files outside the root from the
  container tier. `handles/`, `inputs/`, `.scripts/` and the new `.tether/` are now mounted
  read-only into the container; sandbox-to-harness data travels only through a per-run
  exchange directory read with no-follow, size-capped opens; saved payloads are moved into
  `handles/` by the harness. A replaced protected directory raises `WorkspaceTamperedError`.
- **Behavior change:** scripts can no longer write into `handles/`, `inputs/` or `.scripts/`
  directly (container tier); use `save(id, obj)`. `save()` ids must match
  `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. `write_file` refuses those directories.
- **Behavior change:** the manifest moved to `.tether/manifest.json` (an existing
  `handles/_manifest.json` is migrated on open). `read_document` scratch files moved to
  `.tether/tmp/`.
- **Behavior change:** one-shot roots are `./.tether/runs/run-XXXXXXXX` (atomic allocation),
  no longer `./.tether/sessions/<N>`.
- `HandleStore` is thread-safe and writes atomically; parallel tool calls in one turn no
  longer fail on the manifest.
- File tools hold the session lock, read only regular files, and are bounded; `read_file`
  rejects binary files with a clear error.
- `run_python` output is capped (`SandboxConfig.max_output_bytes`, default 8192 per stream)
  and a large emitted result is stored as a handle. `inspect_handle` is capped, clamps
  `rows` to 100, and `stats=True` now returns footer min/max/null_count instead of
  `describe()`. `publish_handle` converts dataframes inside the sandbox.
- **API change:** `egress.guarded_get` is replaced by `egress.guarded_fetch`, which streams
  with a byte cap, enforces `FetchConfig.total_timeout_s` (120 s), checks each hop's scheme,
  and returns a `FetchedBody`. New `FetchConfig.max_bytes_ceiling` (100 MB) clamps a
  model-supplied `max_bytes`. `read_document` refuses a truncated download.
- AG-UI turns take the conversation lock and count as activity, so `idle_ttl_s` no longer
  reaps an active thread; conversation ids are validated as a single safe path segment.
- `Tether(...)` no longer calls `load_dotenv()` or mutates the config passed in;
  `python-dotenv` is a declared dependency.
- The sandbox image is built for the host's Python `major.minor`.
- CI runs lint and tests on push and pull request; publishing requires them.
```

- [ ] **Step 4: ROADMAP**

In `docs/ROADMAP.md`:

- Change "Last reviewed: 2026-10-01." to "Last reviewed: 2026-10-02."
- Under `## Delivered` add:

```markdown
### Workspace trust zones and bounded I/O
Design in `docs/superpowers/specs/2026-10-02-workspace-trust-zones-design.md`. Closed a
container-tier escape (the harness followed links sandboxed code planted in the shared
root) by making handle storage, inputs, scripts and harness state read-only to the sandbox
and moving the exchange into a per-run directory; made the handle store thread-safe;
bounded every built-in tool result and every harness-side read (process output, files,
parquet previews, HTTP bodies); put AG-UI turns under the conversation lock and idle clock;
added CI.
```

- Delete the `### Control-plane relocation` section and the `### Streaming fetch bodies` section under `## Planned` (both delivered).
- Under `## Smaller carry-overs` delete the bullets "`read_file` has no binary-file handling", "`inspect_handle` calls `describe()` unguarded", and "Bare `ruff check .` also scans agent worktree directories" (lint is clean and CI uses the scoped command). Keep the `max_file_size_mb` and `_OrchestratedSandbox` bullets, and add:

```markdown
- **Parquet parsing in the host process** — the harness still reads the footer and first
  record batch of a file sandboxed code wrote; a hostile data page is decoded whole. Moving
  description into the sandbox (with the host trusting only sizes) would close it.
```

- [ ] **Step 5: README roadmap paragraph**

In `README.md` under `## Roadmap`, replace the sentence beginning "The **security-hardening phase**" with:

```markdown
The **security-hardening phase** (zero-trust handle metadata, container-by-default, egress guard, per-conversation tools) and the **workspace trust zones** work (read-only harness state in the sandbox, per-run exchange directory, bounded tool results and reads) are delivered; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for residuals.
```

- [ ] **Step 6: Check for forbidden references and stale claims**

Run the forbidden-reference grep from the local conventions file (see Global Constraints) over all tracked files.
Expected: no output.
Run: `git grep -n -E 'guarded_get|_manifest\.json|TETHER_EMIT|TETHER_NEW_HANDLES|TETHER_REGISTRY|_documents' -- tether README.md CHANGELOG.md docs/ROADMAP.md examples evals scripts`
Expected: only the CHANGELOG lines that describe the migration (`handles/_manifest.json`, `guarded_get` is replaced) and the legacy-manifest code path in `tether/handles.py`.

- [ ] **Step 7: Final verification and commit**

Run: `uv run ruff check tether tests && uv run pytest -p no:warnings`
Expected: lint clean; all tests pass, with `tests/test_container_live.py` **running** (start the container runtime if it reports skips).
Run: `find tether tests -name '*.bak' -o -name '*.orig'` — expected: no output.
Run: `git status --short` — expected: only the three documentation files modified.

```bash
git add README.md CHANGELOG.md docs/ROADMAP.md
git commit -m "docs: workspace trust zones, bounded I/O, and corrected status claims"
```

---

## Acceptance

The work is done when, on `feat/workspace-trust-zones`:

1. `uv run pytest -p no:warnings` passes with the container-live tests executed, not skipped.
2. `uv run ruff check tether tests` is clean.
3. This reproduction, run against the container tier, prints `result None`, leaves the sentinel unchanged, and stores `h1` correctly:

```python
import json, tempfile
from pathlib import Path
from tether import Session, TetherConfig
from tether.config import SandboxConfig

base = Path(tempfile.mkdtemp(dir=Path.home() / ".tether"))
secret = base / "secret.json"; secret.write_text(json.dumps({"k": "HOST-SECRET"}))
victim = base / "victim.txt"; victim.write_text("original")
s = Session.create(TetherConfig(root_dir=base / "root", sandbox=SandboxConfig(backend="container")))
r = s.sandbox.run_code(
    "import os\n"
    f"os.symlink({str(secret)!r}, os.path.join(os.environ['TETHER_RUN_DIR'], 'emit.json'))\n"
    "for n in ('handles/h1.txt', '.tether/manifest.json'):\n"
    "    try: os.symlink(" + repr(str(victim)) + ", n)\n"
    "    except OSError: pass\n")
print("result", r.result)
h = s.store.put("ATTACKER TEXT", source="fetch")
print(h.id, repr(s.store.get(h.id)), "| victim:", repr(victim.read_text()))
```

   Expected output: `result None` then `h1 'ATTACKER TEXT' | victim: 'original'`. Remove `base` afterwards with `command rm -rf` on that exact path.
