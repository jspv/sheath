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


def test_is_protected_case_variant_on_case_insensitive_fs(tmp_path):
    """On case-insensitive filesystems, must decide by inode identity, not spelling."""
    ensure_layout(tmp_path)
    root = tmp_path.resolve()
    # On case-insensitive FS, HANDLES and handles refer to the same directory
    if (tmp_path / "HANDLES").exists():
        # Case-insensitive filesystem (e.g., macOS APFS)
        assert is_protected(root, root / "HANDLES" / "x")
        assert is_protected(root, root / ".TETHER")
    else:
        # Case-sensitive filesystem
        pytest.skip("case-sensitive filesystem")


def test_is_protected_nonexistent_first_component(tmp_path):
    """A path with non-existent first component (not a protected dir) is not protected."""
    ensure_layout(tmp_path)
    root = tmp_path.resolve()
    assert not is_protected(root, root / "outputs" / "new.txt")


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
