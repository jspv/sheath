import importlib

import pytest

from sheath import SheathConfig, PathEscapesRootError, Session
from sheath.tools.files import write_file
from sheath.tools.search import _search_python, search


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


def _seed(sess):
    write_file(sess, "a.txt", "alpha\nbeta\nGAMMA gamma\n")
    write_file(sess, "sub/b.txt", "delta\nalpha again\n")


def test_search_finds_matches_across_files(tmp_path):
    sess = _session(tmp_path)
    _seed(sess)
    hits = search(sess, "alpha")
    files = {h["file"] for h in hits}
    assert files == {"a.txt", "sub/b.txt"}
    assert all("line" in h and "text" in h for h in hits)


def test_search_can_target_a_single_file(tmp_path):
    sess = _session(tmp_path)
    _seed(sess)
    hits = search(sess, "alpha", path="a.txt")
    assert {h["file"] for h in hits} == {"a.txt"}
    assert hits[0]["line"] == 1


def test_search_ignore_case(tmp_path):
    sess = _session(tmp_path)
    _seed(sess)
    insensitive = search(sess, "gamma", ignore_case=True)
    assert any("GAMMA" in h["text"] for h in insensitive)


def test_search_respects_max_matches(tmp_path):
    sess = _session(tmp_path)
    write_file(sess, "many.txt", "x\n" * 50)
    hits = search(sess, "x", max_matches=5)
    assert len(hits) == 5


def test_search_rejects_path_outside_root(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(PathEscapesRootError):
        search(sess, "x", path="..")


def test_python_fallback_matches_directly(tmp_path):
    # Exercise the fallback regardless of whether rg is installed.
    sess = _session(tmp_path)
    _seed(sess)
    hits = _search_python(sess.root, sess.root, "alpha", ignore_case=False, max_matches=100)
    assert {h["file"] for h in hits} == {"a.txt", "sub/b.txt"}


def test_rg_and_python_backends_agree(tmp_path):
    # The two backends must return identical results (guards env-dependent drift):
    # one hit per matching line, char-offset col, same text.
    from sheath.tools.files import write_file

    sess = _session(tmp_path)
    write_file(sess, "p.txt", "xx xx xx\naaé match here\nnothing\nxx again\n")

    rg_hits = search(sess, "xx")  # rg path (installed on this machine)
    py_hits = _search_python(sess.root, sess.root, "xx", ignore_case=False, max_matches=100)
    assert rg_hits == py_hits

    # Non-ASCII line: col is a CHARACTER offset on both backends.
    rg_m = search(sess, "match")
    py_m = _search_python(sess.root, sess.root, "match", ignore_case=False, max_matches=100)
    assert rg_m == py_m
    assert rg_m[0]["col"] == 4  # 'aaé ' is 4 characters


def test_long_matching_line_is_clipped(tmp_path):
    # Minified-HTML style: a single enormous line. The hit text must be bounded so a
    # search result can't flood the model context.
    from sheath.tools.files import write_file

    sess = _session(tmp_path)
    write_file(sess, "min.html", "x" * 100_000 + "NEEDLE" + "y" * 100_000)
    hits = search(sess, "NEEDLE")
    assert len(hits) == 1
    assert len(hits[0]["text"]) <= 300        # clipped, not 200k chars
    assert "NEEDLE" in hits[0]["text"]         # window is centered on the match
    assert hits[0]["text"].startswith("…")     # truncation markers present


def test_one_hit_per_line_even_with_multiple_matches(tmp_path):
    from sheath.tools.files import write_file

    sess = _session(tmp_path)
    write_file(sess, "m.txt", "xx xx xx\n")
    assert len(search(sess, "xx")) == 1  # one hit for the line, not three


def test_invalid_regex_raises_valueerror(tmp_path):
    from sheath.tools.files import write_file

    sess = _session(tmp_path)
    write_file(sess, "a.txt", "hello\n")
    with pytest.raises(ValueError):
        search(sess, "(unclosed")  # invalid on both rg and Python backends


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
    search_mod = importlib.import_module("sheath.tools.search")  # the package re-exports the function
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
    search_mod = importlib.import_module("sheath.tools.search")  # the package re-exports the function
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None)   # force the fallback
    sess = _session(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("needle outside\n")
    (sess.root / "link.txt").symlink_to(outside)
    os.mkfifo(sess.root / "pipe")
    (sess.root / "real.txt").write_text("x" * 3_000_000 + " needle tail\nneedle here\n")
    hits = search(sess, "needle")
    assert [(h["file"], h["line"]) for h in hits] == [("real.txt", 2)]


def _force_fallback(monkeypatch):
    importlib.import_module("sheath.tools.search")
    import sys
    monkeypatch.setattr(sys.modules["sheath.tools.search"].shutil, "which", lambda name: None)


@pytest.mark.parametrize("glob", ["../*", "/etc/*", "a/../../*", "..\\*"])
def test_search_rejects_escaping_globs(tmp_path, monkeypatch, glob):
    _force_fallback(monkeypatch)
    sess = _session(tmp_path)
    (tmp_path / "secret.txt").write_text("needle top-secret\n")
    with pytest.raises(ValueError, match="invalid glob"):
        search(sess, "needle", glob=glob)


def test_python_search_does_not_follow_a_linked_directory_out_of_the_root(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    sess = _session(tmp_path)
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    (outdir / "s.txt").write_text("needle outside\n")
    (sess.root / "linkdir").symlink_to(outdir)
    assert search(sess, "needle") == []


@pytest.mark.skipif(__import__("shutil").which("rg") is None, reason="ripgrep not on PATH")
def test_rg_search_handles_non_utf8_lines_and_names(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "latin.txt").write_bytes(b"caf\xe9 needle\n")
    hits = search(sess, "needle")
    assert len(hits) == 1 and "needle" in hits[0]["text"] and hits[0]["col"] == 5


def test_search_result_is_capped_by_total_size(tmp_path):
    import json
    sess = _session(tmp_path)
    (sess.root / "big.txt").write_text(("needle " + "y" * 2000 + "\n") * 1000)
    hits = search(sess, "needle", max_matches=1000)
    assert len(json.dumps(hits)) <= 61_000
    assert "more matches omitted" in hits[-1]["note"]


def test_python_fallback_bounds_a_catastrophic_regex(tmp_path, monkeypatch):
    # A model-supplied pattern with exponential backtracking against a model-written file
    # must not pin the host process: the scan runs in a child that is killed on timeout.
    import threading
    import time

    _force_fallback(monkeypatch)
    search_mod = importlib.import_module("sheath.tools.search")
    monkeypatch.setattr(search_mod, "_PY_TIMEOUT_S", 1.0)
    sess = _session(tmp_path)
    (sess.root / "evil.txt").write_text("a" * 30 + "!\n")
    ticks = []
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.05)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    start = time.monotonic()
    try:
        with pytest.raises(ValueError, match="search timed out"):
            search(sess, "(a+)+$")
    finally:
        done.set()
    elapsed = time.monotonic() - start
    assert elapsed < 10
    assert len(ticks) >= 10          # the host kept running Python while the search ran


def test_python_fallback_worker_keeps_confinement_and_results(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    sess = _session(tmp_path)
    _seed(sess)
    hits = search(sess, "alpha", glob="*.txt")
    assert [(h["file"], h["line"], h["col"]) for h in hits] == [("a.txt", 1, 0),
                                                              ("sub/b.txt", 2, 0)]
    with pytest.raises(ValueError, match="invalid search pattern"):
        search(sess, "(unclosed")


@pytest.mark.skipif(__import__("shutil").which("rg") is None, reason="ripgrep not on PATH")
def test_rg_error_text_never_contains_the_host_root(tmp_path):
    import os
    sess = _session(tmp_path)
    locked = sess.root / "sub" / "locked.txt"
    locked.parent.mkdir()
    locked.write_text("needle\n")
    os.chmod(locked, 0)
    try:
        with pytest.raises(ValueError) as ei:
            search(sess, "needle", path="sub")
    finally:
        os.chmod(locked, 0o644)
    assert str(sess.root) not in str(ei.value) and str(tmp_path) not in str(ei.value)
    assert "locked.txt" in str(ei.value)
    (sess.root / "sub" / "ok.txt").write_text("needle\n")
    # rg walks files in parallel, so the order across files is not fixed.
    assert sorted(h["file"] for h in search(sess, "needle", path="sub")) == ["sub/locked.txt",
                                                                             "sub/ok.txt"]
