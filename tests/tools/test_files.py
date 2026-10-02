import os
import threading
import time

import pytest

from tether import TetherConfig, PathEscapesRootError, Session
from tether.tools.files import list_files, read_file, write_file


def _session(tmp_path):
    return Session.create(TetherConfig(root_dir=tmp_path / "r"))


def test_write_then_read_roundtrip(tmp_path):
    sess = _session(tmp_path)
    msg = write_file(sess, "notes/a.txt", "hello\nworld\n")
    assert "a.txt" in msg
    assert read_file(sess, "notes/a.txt") == "hello\nworld\n"


def test_read_is_bounded_by_limit_and_offset(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "big.txt", "".join(f"line{i}\n" for i in range(100)))
    out = read_file(sess, "big.txt", offset=10, limit=3)
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
    assert read_file(sess, "a.txt", offset=100) == ""


def test_read_rejects_negative_offset_or_limit(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "a.txt", "one\ntwo\n")
    with pytest.raises(ValueError):
        read_file(sess, "a.txt", offset=-1)
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
