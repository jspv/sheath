import os

import httpx

import tether.tools.documents as docmod
from tether import TetherConfig, Session
from tether.config import DocumentConfig, FetchConfig
from tether.egress import FetchedBody
from tether.tools.documents import prefetch_models, read_document


def _session(tmp_path, **fetch_kw):
    return Session.create(TetherConfig(root_dir=tmp_path / "r", fetch=FetchConfig(**fetch_kw)))


def _body(url, content, truncated=False):
    return FetchedBody(200, url, httpx.Headers({}), "utf-8", content, truncated)


def _assert_private_snapshot(session, src, suffix, content):
    """The converter got a copy in harness-private scratch, not the workspace file."""
    assert src.startswith(str(session.root / ".tether" / "tmp") + os.sep)
    assert src.endswith(suffix)
    with open(src, "rb") as f:
        assert f.read() == content


def test_path_source_converts_to_markdown_handle(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    summary = read_document(sess, "report.pdf", convert=lambda src: "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert summary["kind"] == "text"
    assert "Title" in summary["preview"]
    assert summary["source"] == "read_document(report.pdf)"
    hid = summary["id"]
    assert "| a | b |" in sess.store.get(hid)


def test_path_is_resolved_under_root_before_conversion(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "sub").mkdir()
    (sess.root / "sub" / "doc.docx").write_bytes(b"x")
    seen = {}

    def fake_convert(src):
        _assert_private_snapshot(sess, src, ".docx", b"x")
        seen["src"] = src
        return "# ok"

    read_document(sess, "sub/doc.docx", convert=fake_convert)
    assert "src" in seen


def test_read_document_blocks_an_internal_url(tmp_path):
    session = _session(tmp_path)
    out = read_document(session, "http://169.254.169.254/latest/meta-data/",
                        convert=lambda src: "should never run")
    assert "error" in out
    assert "blocked by egress policy" in out["error"]


def test_read_document_hands_the_converter_a_local_path_not_a_url(tmp_path, monkeypatch):
    """Docling must never receive the URL: it would follow its own redirects, outside
    the guard. We download through guarded_fetch and convert the local file."""
    session = _session(tmp_path)
    seen = []

    def fake_guarded_fetch(url, cfg, *, client, max_bytes, resolve=None):
        seen.append(url)
        return _body(url, b"%PDF-1.4 fake")

    monkeypatch.setattr("tether.tools.documents.guarded_fetch", fake_guarded_fetch)

    converted = []

    def convert(src: str) -> str:
        converted.append(src)
        return "# Title\n\nbody"

    out = read_document(session, "https://example.com/report.pdf", convert=convert)

    assert out["kind"] == "text"
    assert seen == ["https://example.com/report.pdf"]
    assert len(converted) == 1
    assert not converted[0].startswith("http")           # a local path, never the URL
    assert str(session.root / ".tether" / "tmp") in converted[0]   # in private scratch


def test_read_document_still_converts_a_workspace_path(tmp_path):
    session = _session(tmp_path)
    (session.root / "doc.txt").write_text("hello", encoding="utf-8")
    converted = []

    def convert(src: str) -> str:
        _assert_private_snapshot(session, src, ".txt", b"hello")
        converted.append(src)
        return "# hello"

    out = read_document(session, "doc.txt", convert=convert)
    assert out["kind"] == "text"
    assert len(converted) == 1


def test_read_document_reports_a_download_failure(tmp_path, monkeypatch):
    session = _session(tmp_path)

    def failing(url, cfg, *, client, max_bytes, resolve=None):
        raise httpx.HTTPError("connection reset")

    monkeypatch.setattr("tether.tools.documents.guarded_fetch", failing)
    out = read_document(session, "https://example.com/x.pdf", convert=lambda s: "")
    assert "could not download" in out["error"]


def test_read_document_caps_the_downloaded_body_at_max_bytes(tmp_path, monkeypatch):
    """The download is read with FetchConfig.max_bytes as its cap, and a capped (partial)
    document is refused rather than converted."""
    session = _session(tmp_path, max_bytes=10)
    caps = []

    def fake(url, cfg, *, client, max_bytes, resolve=None):
        caps.append(max_bytes)
        return _body(url, b"x" * max_bytes, truncated=True)

    monkeypatch.setattr("tether.tools.documents.guarded_fetch", fake)
    sizes = []
    out = read_document(session, "https://example.com/big.pdf",
                        convert=lambda src: sizes.append(os.path.getsize(src)) or "# md")
    assert caps == [10]
    assert "larger than" in out["error"]
    assert sizes == []
    scratch = session.root / ".tether" / "tmp"
    assert not scratch.exists() or list(scratch.iterdir()) == []


def test_read_document_closes_its_client_on_every_path(tmp_path, monkeypatch):
    session = _session(tmp_path)
    clients = []
    real_client = httpx.Client

    def tracking_client(*a, **kw):
        c = real_client(*a, **kw)
        clients.append(c)
        return c

    monkeypatch.setattr("tether.tools.documents.httpx.Client", tracking_client)

    def failing(url, cfg, *, client, max_bytes, resolve=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr("tether.tools.documents.guarded_fetch", failing)
    read_document(session, "https://example.com/x.pdf", convert=lambda s: "")
    assert len(clients) == 1 and clients[0].is_closed


def _stub_get(monkeypatch, content=b"%PDF-1.4 fake"):
    monkeypatch.setattr("tether.tools.documents.guarded_fetch",
                        lambda url, cfg, *, client, max_bytes, resolve=None:
                        _body(url, content))


def test_downloaded_documents_are_not_listed_as_artifacts(tmp_path, monkeypatch):
    session = _session(tmp_path)
    (session.root / "report.txt").write_text("mine", encoding="utf-8")
    from pathlib import Path

    _stub_get(monkeypatch)
    seen = []

    def convert(src):
        seen.append(src)
        assert Path(src).exists()       # the download really happened, as a local file
        return "# md"

    out = read_document(session, "https://example.com/a.pdf", convert=convert)
    assert out["kind"] == "text"
    assert session.artifacts == ["report.txt"]
    assert seen and seen[0].startswith(str(session.root / ".tether" / "tmp") + os.sep)


def test_overlong_url_suffix_does_not_crash_and_lands_as_bin(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _stub_get(monkeypatch)
    seen = []
    out = read_document(session, "https://example.com/doc." + "a" * 300,
                        convert=lambda src: seen.append(src) or "# md")
    assert out["kind"] == "text"
    assert seen[0].endswith(".bin")


def test_ordinary_suffix_is_preserved(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _stub_get(monkeypatch)
    seen = []
    read_document(session, "https://example.com/a/b.docx?x=1",
                  convert=lambda src: seen.append(src) or "# md")
    assert seen[0].endswith(".docx")


def test_filesystem_failure_during_download_is_a_structured_error(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _stub_get(monkeypatch)

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("tether.tools.documents.tempfile.mkstemp", boom)
    out = read_document(session, "https://example.com/a.pdf", convert=lambda s: "")
    assert "could not store" in out["error"]


def test_path_escape_returns_structured_error(tmp_path):
    sess = _session(tmp_path)
    out = read_document(sess, "../../etc/passwd", convert=lambda src: "nope")
    assert "error" in out and "escape" in out["error"].lower()
    assert out["source"] == "../../etc/passwd"
    assert not sess.handles


def test_unknown_scheme_returns_structured_error(tmp_path):
    sess = _session(tmp_path)
    out = read_document(sess, "ftp://host/f.pdf", convert=lambda src: "nope")
    assert "error" in out and "scheme" in out["error"].lower()
    assert not sess.handles


def test_conversion_failure_returns_structured_error(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "broken.pdf").write_bytes(b"x")

    def boom(src):
        raise ValueError("corrupt pdf")

    out = read_document(sess, "broken.pdf", convert=boom)
    assert "error" in out and "could not read document" in out["error"].lower()
    assert "corrupt pdf" in out["error"]
    assert out["source"] == "broken.pdf"
    assert not sess.handles


def test_missing_docling_returns_actionable_error(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "x.pdf").write_bytes(b"x")

    def not_installed(src):
        raise ModuleNotFoundError("No module named 'docling'")

    out = read_document(sess, "x.pdf", convert=not_installed)
    assert "error" in out and "docling" in out["error"].lower()
    assert "--extra docling" in out["error"]
    assert not sess.handles


def test_default_converter_uses_config_ocr_off_by_default(tmp_path, monkeypatch):
    sess = _session(tmp_path)                                   # default DocumentConfig: ocr=False
    (sess.root / "f.pdf").write_bytes(b"x")
    captured = {}

    def fake_docling(src, ocr):
        _assert_private_snapshot(sess, src, ".pdf", b"x")
        captured.update(src=src, ocr=ocr)
        return "# md"

    monkeypatch.setattr(docmod, "_docling_convert", fake_docling)
    out = read_document(sess, "f.pdf")                          # no injected converter -> default path
    assert out["kind"] == "text"
    assert captured["ocr"] is False


def test_default_converter_enables_ocr_when_configured(tmp_path, monkeypatch):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r", documents=DocumentConfig(ocr=True)))
    (sess.root / "f.pdf").write_bytes(b"x")
    captured = {}
    monkeypatch.setattr(docmod, "_docling_convert",
                        lambda src, ocr: captured.update(ocr=ocr) or "# md")
    read_document(sess, "f.pdf")
    assert captured["ocr"] is True


def test_prefetch_models_skips_ocr_models_by_default():
    calls = {}
    prefetch_models(downloader=lambda **kw: calls.update(kw))
    assert calls["with_rapidocr"] is False                      # matches the OCR-off default


def test_prefetch_models_includes_ocr_models_when_requested():
    calls = {}
    prefetch_models(downloader=lambda **kw: calls.update(kw), ocr=True)
    assert calls["with_rapidocr"] is True


def test_downloaded_temp_file_is_removed_after_a_successful_conversion(tmp_path, monkeypatch):
    """Every read_document(url) wrote up to max_bytes into the session root and left it
    there: a model looping on read_document had an unbounded disk-growth primitive."""
    session = _session(tmp_path)
    _stub_get(monkeypatch)
    for _ in range(3):
        out = read_document(session, "https://example.com/a.pdf", convert=lambda s: "# md")
        assert out["kind"] == "text"
    assert list((session.root / ".tether" / "tmp").iterdir()) == []


def test_downloaded_temp_file_is_removed_after_a_failed_conversion(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _stub_get(monkeypatch)

    def boom(src):
        raise ValueError("corrupt pdf")

    out = read_document(session, "https://example.com/a.pdf", convert=boom)
    assert "could not read document" in out["error"]
    assert list((session.root / ".tether" / "tmp").iterdir()) == []


def test_a_workspace_file_is_never_deleted_by_the_cleanup(tmp_path):
    """Only the scratch download is reaped -- a user's own file is converted in place."""
    session = _session(tmp_path)
    (session.root / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    read_document(session, "report.pdf", convert=lambda s: "# md")
    assert (session.root / "report.pdf").exists()


def test_narrowing_allowed_schemes_is_honored_by_read_document(tmp_path, monkeypatch):
    """allowed_schemes was hard-coded here, so tightening it bound fetch_url and not this."""
    session = _session(tmp_path, allowed_schemes=("https",))

    def must_not_fetch(*a, **kw):
        raise AssertionError("read_document fetched over a scheme the config disallows")

    monkeypatch.setattr("tether.tools.documents.guarded_fetch", must_not_fetch)

    out = read_document(session, "http://example.com/a.pdf", convert=lambda s: "# md")
    assert "unsupported source scheme" in out["error"]
    assert "https" in out["error"]                        # says what is allowed


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


def test_read_document_snapshot_never_follows_a_swapped_in_symlink(tmp_path, monkeypatch):
    """safe_path resolves symlinks, but sandboxed code could swap one in after the check.
    The snapshot opens the checked name once with O_NOFOLLOW, so a symlink there is refused.
    Simulated by a safe_path that returns the name unresolved."""
    sess = _session(tmp_path)
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF-secret")
    (sess.root / "link.pdf").symlink_to(outside)
    monkeypatch.setattr(docmod, "safe_path", lambda root, p: root / p)
    out = read_document(sess, "link.pdf",
                        convert=lambda p: (_ for _ in ()).throw(AssertionError("no convert")))
    assert "symlink" in out["error"]
