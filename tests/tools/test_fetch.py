import httpx

from tether import TetherConfig, Session
from tether.config import FetchConfig
from tether.tools.fetch import fetch_url


def _session(tmp_path, **fetch_kw):
    return Session.create(TetherConfig(root_dir=tmp_path / "r",
                                       fetch=FetchConfig(**fetch_kw)))


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_json_returns_json_handle(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"hello": "world"})

    summary = fetch_url(sess, "https://example.com/data", client=_client(handler))
    assert summary["kind"] == "json"
    assert sess.store.get(summary["id"]) == {"hello": "world"}


def test_fetch_text_returns_text_handle(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, text="plain body", headers={"content-type": "text/plain"})

    summary = fetch_url(sess, "https://example.com/page", client=_client(handler))
    assert summary["kind"] == "text"
    assert sess.store.get(summary["id"]) == "plain body"


def test_fetch_rejects_disallowed_scheme(tmp_path):
    """Structured, not raised: every other failure in this tool returns a dict the model
    can read and adapt to, and sibling read_document returns one for exactly this case."""
    sess = _session(tmp_path)
    out = fetch_url(sess, "file:///etc/passwd")
    assert "scheme" in out["error"] and "not allowed" in out["error"]
    assert out["url"] == "file:///etc/passwd"
    assert out["status"] is None


def test_fetch_truncates_body_over_max_bytes(tmp_path):
    # An over-large body is truncated (and flagged), not a hard failure -- the agent can
    # still search/inspect what came back.
    cfg = TetherConfig(root_dir=tmp_path / "r")
    cfg.fetch.max_bytes = 5
    sess = Session.create(cfg)

    def handler(request):
        return httpx.Response(200, text="this is way too long", headers={"content-type": "text/plain"})

    out = fetch_url(sess, "https://example.com/big", client=_client(handler))
    assert out["truncated"] is True
    assert out["full_bytes"] == len("this is way too long")
    assert sess.store.get(out["id"]) == "this "  # first 5 bytes


def test_fetch_returns_structured_error_on_http_error(tmp_path):
    # 4xx/5xx is data the model can adapt to, not a crash.
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(403, text="forbidden")

    out = fetch_url(sess, "https://example.com/blocked", client=_client(handler))
    assert out["status"] == 403
    assert "error" in out
    assert out["url"] == "https://example.com/blocked"
    assert sess.store.manifest_handles() == {}  # nothing stored on error


def test_fetch_returns_structured_error_on_network_failure(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        raise httpx.ConnectError("boom")

    out = fetch_url(sess, "https://example.com/down", client=_client(handler))
    assert out["status"] is None
    assert "request failed" in out["error"]


def test_fetch_follows_redirects(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://example.com/new"})
        return httpx.Response(200, text="final page", headers={"content-type": "text/plain"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    summary = fetch_url(sess, "https://example.com/old", client=client)
    assert summary["kind"] == "text"
    assert sess.store.get(summary["id"]) == "final page"


def test_default_client_has_user_agent_and_does_not_follow_redirects(tmp_path):
    """_default_client sets follow_redirects=False so guarded_fetch can handle redirects
    with re-validation against egress policy for each hop."""
    from tether.tools.fetch import _default_client

    c = _default_client(TetherConfig().fetch)
    try:
        assert c.follow_redirects is False
        assert "Mozilla" in c.headers["user-agent"]
    finally:
        c.close()


_HTML_PAGE = (
    "<html><head><title>T</title></head><body>"
    "<nav>Home About Contact Login Signup</nav>"
    "<article><h1>Pricing Details</h1>"
    "<p>Our flagship model costs 5 dollars per million input tokens.</p>"
    "<p>Output tokens are billed at 25 dollars per million.</p>"
    "<p>Batch processing gives a 50 percent discount on all usage.</p></article>"
    "<footer>Copyright 2026. Cookie banner here. Terms of service.</footer></body></html>"
)


def test_fetch_html_is_converted_to_clean_markdown(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, text=_HTML_PAGE, headers={"content-type": "text/html"})

    summary = fetch_url(sess, "https://example.com/pricing", client=_client(handler))
    body = sess.store.get(summary["id"])
    assert "flagship model costs 5 dollars" in body
    assert "Cookie banner" not in body
    assert "Signup" not in body
    assert "<article>" not in body


def test_fetch_html_raw_flag_keeps_raw(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, text=_HTML_PAGE, headers={"content-type": "text/html"})

    summary = fetch_url(sess, "https://example.com/pricing", raw=True, client=_client(handler))
    assert "<article>" in sess.store.get(summary["id"])


def test_fetch_non_html_text_unchanged(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, text="just plain text", headers={"content-type": "text/plain"})

    summary = fetch_url(sess, "https://example.com/p.txt", client=_client(handler))
    assert sess.store.get(summary["id"]) == "just plain text"


def test_fetch_binary_xls_is_stored_intact(tmp_path):
    # An .xls (OLE2 magic) must be stored as raw bytes, not mangled into a text handle.
    sess = _session(tmp_path)
    xls_bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00\x01\x02\x03" * 50

    def handler(request):
        return httpx.Response(200, content=xls_bytes,
                              headers={"content-type": "application/vnd.ms-excel"})

    summary = fetch_url(sess, "http://x.com/data.xls", client=_client(handler))
    assert summary["kind"] == "binary"
    assert summary["path"].endswith(".xls")          # extension preserved from the URL
    from pathlib import Path
    assert Path(sess.store.get(summary["id"])).read_bytes() == xls_bytes  # bytes intact


def test_fetch_binary_detected_by_content_sniff(tmp_path):
    # Generic content-type, but the bytes don't decode as utf-8 -> treat as binary.
    sess = _session(tmp_path)
    blob = bytes(range(256)) * 4  # not valid utf-8

    def handler(request):
        return httpx.Response(200, content=blob, headers={"content-type": "application/octet-stream"})

    summary = fetch_url(sess, "http://x.com/blob", client=_client(handler))
    assert summary["kind"] == "binary"


def test_fetch_url_returns_structured_error_for_internal_address(tmp_path):
    """Blocked, and reported the way a network failure is -- so the agent can adapt."""
    session = _session(tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="should never be reached")),
        follow_redirects=False)
    with client:
        out = fetch_url(session, "http://10.0.0.1/admin", client=client)
    assert out["status"] is None
    assert "blocked by egress policy" in out["error"]   # the wrapper fetch_url adds
    assert "blocked internal address" in out["error"]   # the reason egress gave
    assert out["url"] == "http://10.0.0.1/admin"


def test_fetch_url_blocks_cloud_metadata_with_its_own_reason(tmp_path):
    """Metadata endpoints raise a DIFFERENT message than ordinary internal addresses
    ("blocked cloud metadata endpoint", not "blocked internal address"), because they are
    denied unconditionally ahead of the allowlist. Asserting the generic wording here would
    pass for the wrong reason, or fail spuriously."""
    session = _session(tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="should never be reached")),
        follow_redirects=False)
    with client:
        out = fetch_url(session, "http://169.254.169.254/latest/meta-data/", client=client)
    assert out["status"] is None
    assert "blocked cloud metadata endpoint" in out["error"]


def test_fetch_url_allows_an_allowlisted_internal_host(tmp_path):
    session = _session(tmp_path, allow_private_hosts=("127.0.0.1",))
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="internal report",
                                 headers={"content-type": "text/plain"})),
        follow_redirects=False)
    with client:
        out = fetch_url(session, "http://127.0.0.1/report", client=client)
    assert "error" not in out
    assert out["kind"] == "text"


def test_fetch_url_blocks_a_redirect_into_an_internal_address(tmp_path):
    """The guard must apply to hops, not just the initial URL -- and fetch_url must report
    it with the same structured-error contract as any other failure."""
    session = _session(tmp_path)

    def handler(request):
        if "127.0.0.1" in str(request.url):
            return httpx.Response(200, text="internal data")
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8080/admin"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with client:
        out = fetch_url(session, "http://93.184.216.34/start", client=client)

    assert out["status"] is None
    assert "blocked by egress policy" in out["error"]
    assert "127.0.0.1" in out["error"]


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


def test_fetch_url_survives_an_unknown_charset(tmp_path):
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, content=b"plain",
                              headers={"content-type": "text/plain; charset=x-nonsense"})

    summary = fetch_url(sess, "https://example.com/x", client=_client(handler))
    assert sess.store.get(summary["id"]) == "plain"


def test_fetch_url_survives_a_charset_that_cannot_decode_with_replace(tmp_path):
    """idna is a real codec, so httpx keeps it, but it rejects errors='replace' with a
    UnicodeError; that must not escape the structured-error contract."""
    sess = _session(tmp_path)

    def handler(request):
        return httpx.Response(200, content=b"plain",
                              headers={"content-type": "text/plain; charset=idna"})

    summary = fetch_url(sess, "https://example.com/x", client=_client(handler))
    assert sess.store.get(summary["id"]) == "plain"
