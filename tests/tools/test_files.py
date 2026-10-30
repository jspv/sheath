import os
import threading
import time

import pytest

from sheath import SheathConfig, PathEscapesRootError, Session
from sheath.tools.files import list_files, read_file, write_file


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


def test_write_then_read_roundtrip(tmp_path):
    sess = _session(tmp_path)
    msg = write_file(sess, "notes/a.txt", "hello\nworld\n")
    assert "a.txt" in msg
    assert read_file(sess, "notes/a.txt") == "hello\nworld\n"


def test_read_is_bounded_by_limit_and_start_line(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "big.txt", "".join(f"line{i}\n" for i in range(1, 101)))
    out = read_file(sess, "big.txt", start_line=10, limit=3)
    assert out == "line10\nline11\nline12\n"


def test_read_file_caps_total_characters_on_long_lines(tmp_path):
    # Minified-HTML style: one enormous line. limit=lines doesn't bound chars, so the
    # char cap must kick in to protect the context window.
    sess = _session(tmp_path)
    write_file(sess, "big.html", "x" * 200_000)  # single 200k-char line
    out = read_file(sess, "big.html")
    assert len(out) <= 50_000 + 200  # cap + truncation note
    assert "truncated" in out.lower()


def test_read_past_eof_returns_empty(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "one\ntwo\n")
    assert read_file(sess, "a.txt", start_line=100) == ""


def test_read_rejects_start_line_below_one_or_negative_limit(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "one\ntwo\n")
    with pytest.raises(ValueError, match="1-based"):
        read_file(sess, "a.txt", start_line=0)
    with pytest.raises(ValueError):
        read_file(sess, "a.txt", limit=-1)


def test_write_rejects_path_outside_root(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(PathEscapesRootError):
        write_file(sess, "../escape.txt", "x")


def test_read_rejects_path_outside_root(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(PathEscapesRootError):
        read_file(sess, "/etc/passwd")


def test_list_files_empty_path_means_root(tmp_path):
    # A model passing "" for "current dir" should get the root listing, not an error.
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "1")
    assert "a.txt" in list_files(sess, "")


def test_list_files_returns_relative_paths(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "1")
    write_file(sess, "sub/b.txt", "2")
    listing = set(list_files(sess, "."))
    assert "a.txt" in listing
    assert "sub/b.txt" in listing


def test_list_files_rejects_path_outside_root(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(PathEscapesRootError):
        list_files(sess, "..")



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
                                    ".sheath/manifest.json"])
def test_write_file_refuses_protected_directories(tmp_path, target):
    sess = _session(tmp_path)
    with pytest.raises(ValueError, match="managed by sheath"):
        write_file(sess, target, "x")


def test_write_file_refuses_a_link_into_a_protected_directory(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "alias").symlink_to(sess.root / "handles")
    with pytest.raises(ValueError, match="managed by sheath"):
        write_file(sess, "alias/h9.txt", "x")


def test_list_files_skips_state_dir_and_links_and_is_capped(tmp_path):
    sess = _session(tmp_path)
    sess.store.put("x", source="t")
    (sess.root / "a.txt").write_text("1")
    (sess.root / "link").symlink_to(sess.root / "a.txt")
    listed = list_files(sess)
    assert "a.txt" in listed and "handles/h1.txt" in listed
    assert "link" not in listed and not any(p.startswith(".sheath") for p in listed)
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


def test_read_file_refuses_a_symlink_swapped_in_after_the_check(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("top-secret\n")
    (sess.root / "link.txt").symlink_to(outside)
    monkeypatch.setattr("sheath.tools.files.safe_path", lambda root, p: root / p)
    with pytest.raises(ValueError, match="symlink"):
        read_file(sess, "link.txt")


def test_write_file_refuses_a_symlink_swapped_in_after_the_check(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n")
    (sess.root / "link.txt").symlink_to(outside)
    monkeypatch.setattr("sheath.tools.files.safe_path", lambda root, p: root / p)
    with pytest.raises(ValueError, match="symlink"):
        write_file(sess, "link.txt", "pwned")
    assert outside.read_text() == "original\n"


def test_write_file_to_a_fifo_fails_promptly(tmp_path):
    sess = _session(tmp_path)
    os.mkfifo(sess.root / "pipe")
    with pytest.raises(ValueError, match="not a regular file"):
        write_file(sess, "pipe", "x")


def test_write_file_to_a_directory_is_refused(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "d").mkdir()
    with pytest.raises(ValueError):
        write_file(sess, "d", "x")


def test_read_file_splits_bare_carriage_return_lines(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "mac.txt").write_bytes(b"one\rtwo\rthree\r")
    assert read_file(sess, "mac.txt", start_line=2, limit=1) == "two\r"
    assert read_file(sess, "mac.txt", start_line=3, limit=5) == "three\r"


@pytest.mark.parametrize("width", [65535, 131071])
def test_read_file_counts_crlf_lines_once_at_a_chunk_boundary(tmp_path, width):
    sess = _session(tmp_path)
    line = b"y" * width + b"\r\n"
    (sess.root / "dos.txt").write_bytes(line * 3 + b"tail\r\n")
    big = 10 * width
    assert read_file(sess, "dos.txt", start_line=2, limit=1, max_chars=big) == line.decode()
    assert read_file(sess, "dos.txt", start_line=4, limit=5, max_chars=big) == "tail\r\n"
    assert read_file(sess, "dos.txt", limit=100, max_chars=big) == (
        (line * 3 + b"tail\r\n").decode())


def test_write_file_with_unencodable_content_leaves_the_file_intact(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "keep.txt", "precious")
    with pytest.raises(UnicodeEncodeError):
        write_file(sess, "keep.txt", "bad \ud800 surrogate")
    assert (sess.root / "keep.txt").read_text() == "precious"


# --- read_file line_numbers -------------------------------------------------------------

def test_read_file_line_numbers_prefix_each_line(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "one\ntwo\nthree\n")
    assert read_file(sess, "a.txt", start_line=2, line_numbers=True) == (
        "     2\ttwo\n     3\tthree\n")


# --- list_files pattern -----------------------------------------------------------------

def _tree(sess):
    for rel in ("a.csv", "b.txt", "data/c.csv", "data/deep/d.csv", "data/e.json"):
        write_file(sess, rel, "x")


def test_list_files_pattern_star_stays_in_one_folder(tmp_path):
    sess = _session(tmp_path)
    _tree(sess)
    assert list_files(sess, ".", "*.csv") == ["a.csv"]


def test_list_files_pattern_double_star_spans_folders(tmp_path):
    sess = _session(tmp_path)
    _tree(sess)
    assert list_files(sess, ".", "**/*.csv") == ["a.csv", "data/c.csv", "data/deep/d.csv"]


def test_list_files_pattern_is_relative_to_path(tmp_path):
    sess = _session(tmp_path)
    _tree(sess)
    assert list_files(sess, "data", "*.json") == ["data/e.json"]
    assert list_files(sess, "data", "deep/*") == ["data/deep/d.csv"]


def test_list_files_pattern_on_a_single_file(tmp_path):
    sess = _session(tmp_path)
    _tree(sess)
    assert list_files(sess, "b.txt", "*.txt") == ["b.txt"]
    assert list_files(sess, "b.txt", "*.csv") == []


@pytest.mark.parametrize("bad", ["/etc/*", "../*", "data/../../x"])
def test_list_files_rejects_escaping_patterns(tmp_path, bad):
    sess = _session(tmp_path)
    with pytest.raises(ValueError, match="invalid pattern"):
        list_files(sess, ".", bad)


# --- edit_file --------------------------------------------------------------------------

from sheath.tools.files import edit_file  # noqa: E402


def test_edit_file_replaces_a_unique_match_and_shows_numbered_lines(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.py", "x = 1\ny = 2\nz = 3\n")
    out = edit_file(sess, "a.py", "y = 2", "y = 20")
    assert read_file(sess, "a.py") == "x = 1\ny = 20\nz = 3\n"
    assert out.startswith("replaced 1 occurrence(s) in a.py")
    assert "     2\ty = 20\n" in out


def test_edit_file_refuses_an_ambiguous_match_and_names_the_lines(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "dup\nother\ndup\n")
    with pytest.raises(ValueError, match=r"occurs 2 times.*lines 1, 3"):
        edit_file(sess, "a.txt", "dup", "new")
    assert read_file(sess, "a.txt") == "dup\nother\ndup\n"


def test_edit_file_replace_all(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "dup\nother\ndup\n")
    assert edit_file(sess, "a.txt", "dup", "new", replace_all=True).startswith(
        "replaced 2 occurrence(s)")
    assert read_file(sess, "a.txt") == "new\nother\nnew\n"


def test_edit_file_can_delete_text(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "keep\ndrop\nkeep2\n")
    edit_file(sess, "a.txt", "drop\n", "")
    assert read_file(sess, "a.txt") == "keep\nkeep2\n"


def test_edit_file_no_match_points_at_a_near_match(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "one\nHello World\n")
    with pytest.raises(ValueError, match="case-insensitive match starts on line 2"):
        edit_file(sess, "a.txt", "hello world", "x")
    with pytest.raises(ValueError, match="differing only in whitespace starts on line 2"):
        edit_file(sess, "a.txt", "Hello   World", "x")
    with pytest.raises(ValueError, match="not found"):
        edit_file(sess, "a.txt", "absent", "x")


@pytest.mark.parametrize("old,new", [("", "x"), ("same", "same")])
def test_edit_file_rejects_empty_or_identical_text(tmp_path, old, new):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "same\n")
    with pytest.raises(ValueError):
        edit_file(sess, "a.txt", old, new)


def test_edit_file_keeps_crlf_line_endings_and_permissions(tmp_path):
    sess = _session(tmp_path)
    target = sess.root / "dos.txt"
    target.write_bytes(b"a\r\nb\r\nc\r\n")
    os.chmod(target, 0o640)
    out = edit_file(sess, "dos.txt", "b", "B")
    assert target.read_bytes() == b"a\r\nB\r\nc\r\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert "     2\tB\r\n" in out
    assert [p.name for p in sess.root.iterdir() if p.name.endswith(".tmp")] == []


def test_edit_file_line_numbers_match_read_file_with_bare_cr(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "mac.txt").write_bytes(b"one\rtwo\rtwo\r")
    with pytest.raises(ValueError, match="lines 2, 3"):
        edit_file(sess, "mac.txt", "two", "x")


@pytest.mark.parametrize("rel", ["handles/x.txt", "inputs/x.txt", ".sheath/x.txt"])
def test_edit_file_refuses_protected_directories(tmp_path, rel):
    sess = _session(tmp_path)
    target = sess.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data\n")
    with pytest.raises(ValueError, match="managed by sheath"):
        edit_file(sess, rel, "data", "evil")
    assert target.read_text() == "data\n"


def test_edit_file_refuses_escaping_links_binary_and_non_utf8(tmp_path):
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("data\n")
    os.symlink(outside, sess.root / "link.txt")
    with pytest.raises(PathEscapesRootError):
        edit_file(sess, "link.txt", "data", "x")
    assert outside.read_text() == "data\n"
    (sess.root / "bin.dat").write_bytes(b"data\0\0")
    with pytest.raises(ValueError, match="binary"):
        edit_file(sess, "bin.dat", "data", "x")
    (sess.root / "latin.txt").write_bytes(b"caf\xe9 data\n")
    with pytest.raises(ValueError, match="UTF-8"):
        edit_file(sess, "latin.txt", "data", "x")


def test_edit_file_refuses_files_over_the_size_cap(tmp_path, monkeypatch):
    import sheath.tools.files as files
    monkeypatch.setattr(files, "_MAX_EDIT_BYTES", 10)
    sess = _session(tmp_path)
    write_file(sess, "big.txt", "data " * 10)
    with pytest.raises(ValueError, match="run_python"):
        edit_file(sess, "big.txt", "data", "x")


def test_edit_file_rejects_path_outside_root(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(PathEscapesRootError):
        edit_file(sess, "../escape.txt", "a", "b")


def test_edit_file_preview_is_bounded(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "start\n")
    out = edit_file(sess, "a.txt", "start", "\n".join("y" * 200 for _ in range(500)))
    assert len(out) <= 4200
    assert "preview truncated" in out
