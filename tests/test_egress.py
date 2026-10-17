import httpx
import pytest

from sheath.config import FetchConfig
from sheath.egress import (
    BlockedAddressError,
    FetchDeadlineExceeded,
    guarded_fetch,
    validate_url,
)


def _resolver(mapping):
    """Stub DNS: hostname -> list of addresses. No network in tests."""
    def resolve(host: str) -> list[str]:
        return mapping[host]
    return resolve


@pytest.mark.parametrize("addr,label", [
    ("127.0.0.1", "ipv4 loopback"),
    ("::1", "ipv6 loopback"),
    ("::ffff:127.0.0.1", "ipv4-mapped loopback"),
    ("10.0.0.5", "private 10/8"),
    ("172.16.0.1", "private 172.16/12"),
    ("192.168.1.1", "private 192.168/16"),
    ("fd00::1", "ipv6 unique-local"),
    ("169.254.169.254", "cloud metadata"),
    ("169.254.1.1", "link-local"),
    ("0.0.0.0", "unspecified"),
    ("224.0.0.1", "multicast"),
    ("100.64.0.1", "RFC6598 CGNAT — no standard flag catches this one"),
])
def test_validate_url_rejects_internal_addresses(addr, label):
    cfg = FetchConfig()
    with pytest.raises(BlockedAddressError):
        validate_url("http://target.example/x", cfg,
                     resolve=_resolver({"target.example": [addr]}))


@pytest.mark.parametrize("addr", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_validate_url_allows_public_addresses(addr):
    validate_url("https://example.com/x", FetchConfig(),
                 resolve=_resolver({"example.com": [addr]}))


def test_any_denied_address_blocks_the_host():
    """A hostname resolving to both a public and a private address is denied -- the
    attacker picks which one connect() uses, so one bad answer is enough."""
    with pytest.raises(BlockedAddressError):
        validate_url("https://split.example/x", FetchConfig(),
                     resolve=_resolver({"split.example": ["93.184.216.34", "127.0.0.1"]}))


def test_allowlist_admits_a_named_host():
    cfg = FetchConfig(allow_private_hosts=("intranet.corp",))
    validate_url("http://intranet.corp/report", cfg,
                 resolve=_resolver({"intranet.corp": ["10.1.2.3"]}))


def test_allowlist_is_case_insensitive():
    cfg = FetchConfig(allow_private_hosts=("Intranet.Corp",))
    validate_url("http://intranet.corp/x", cfg,
                 resolve=_resolver({"intranet.corp": ["10.1.2.3"]}))


def test_allowlist_admits_a_cidr():
    cfg = FetchConfig(allow_private_hosts=("10.1.0.0/16",))
    validate_url("http://db.internal/x", cfg,
                 resolve=_resolver({"db.internal": ["10.1.2.3"]}))


def test_cidr_allowlist_does_not_admit_other_private_space():
    cfg = FetchConfig(allow_private_hosts=("10.1.0.0/16",))
    with pytest.raises(BlockedAddressError):
        validate_url("http://other.internal/x", cfg,
                     resolve=_resolver({"other.internal": ["10.9.9.9"]}))


def test_empty_allowlist_admits_nothing_internal():
    assert FetchConfig().allow_private_hosts == ()
    with pytest.raises(BlockedAddressError):
        validate_url("http://x.internal/x", FetchConfig(),
                     resolve=_resolver({"x.internal": ["10.0.0.1"]}))


def test_literal_ip_url_is_checked_without_dns():
    """A URL with a bare IP must be validated too, and must not need a resolver."""
    def must_not_resolve(host):
        raise AssertionError(f"resolver called for literal IP {host!r}")

    with pytest.raises(BlockedAddressError):
        validate_url("http://127.0.0.1:8080/admin", FetchConfig(), resolve=must_not_resolve)
    validate_url("http://93.184.216.34/", FetchConfig(), resolve=must_not_resolve)


def test_unresolvable_host_is_blocked():
    def resolve(host):
        raise OSError("name resolution failed")
    with pytest.raises(BlockedAddressError, match="could not resolve"):
        validate_url("http://nope.invalid/x", FetchConfig(), resolve=resolve)


def test_url_without_host_is_blocked():
    with pytest.raises(BlockedAddressError, match="no host"):
        validate_url("http:///x", FetchConfig())


def test_resolver_returning_empty_list_is_blocked():
    """A resolver returning no addresses is a denial, not a pass-through."""
    with pytest.raises(BlockedAddressError, match="no addresses"):
        validate_url("http://nope.example/x", FetchConfig(),
                     resolve=_resolver({"nope.example": []}))


def test_unicode_error_on_resolution_is_blocked():
    """IDNA encoding errors (e.g., over-long labels) raise UnicodeError, which must be caught."""
    def resolve_unicode_error(host):
        raise UnicodeError("label too long")
    with pytest.raises(BlockedAddressError, match="could not resolve"):
        validate_url("http://toolonglabel.example/x", FetchConfig(), resolve=resolve_unicode_error)


@pytest.mark.parametrize("metadata_addr,label", [
    ("169.254.169.254", "AWS/GCP/Azure IMDS"),
    ("169.254.170.2", "AWS ECS task credentials"),
    ("168.63.129.16", "Azure wireserver (public)"),
    ("100.100.100.200", "Alibaba Cloud (in CGNAT)"),
    ("192.0.0.192", "Oracle Cloud"),
    ("fd00:ec2::254", "AWS IMDS IPv6"),
])
def test_metadata_endpoints_are_denied(metadata_addr, label):
    """All known cloud metadata endpoints are denied by default."""
    with pytest.raises(BlockedAddressError, match="blocked cloud metadata endpoint"):
        validate_url("http://target.example/x", FetchConfig(),
                     resolve=_resolver({"target.example": [metadata_addr]}))


@pytest.mark.parametrize("metadata_addr", [
    "169.254.169.254",
    "169.254.170.2",
    "168.63.129.16",
    "100.100.100.200",
    "192.0.0.192",
    "fd00:ec2::254",
])
def test_metadata_endpoints_cannot_be_allowlisted_by_hostname(metadata_addr):
    """Metadata endpoints cannot be overridden by hostname allowlist."""
    # Attempt to allowlist the hostname; metadata check happens before allowlist
    cfg = FetchConfig(allow_private_hosts=("metadata.example",))
    with pytest.raises(BlockedAddressError, match="blocked cloud metadata endpoint"):
        validate_url("http://metadata.example/x", cfg,
                     resolve=_resolver({"metadata.example": [metadata_addr]}))


def test_metadata_endpoints_cannot_be_allowlisted_by_cidr():
    """Metadata endpoints cannot be overridden by CIDR allowlist."""
    # Attempt to allowlist the 169.254.0.0/16 range containing AWS IMDS
    cfg = FetchConfig(allow_private_hosts=("169.254.0.0/16",))
    with pytest.raises(BlockedAddressError, match="blocked cloud metadata endpoint"):
        validate_url("http://target.example/x", cfg,
                     resolve=_resolver({"target.example": ["169.254.169.254"]}))


def test_allowlisted_host_with_mixed_private_addresses():
    """A host resolving to both allowed (in CIDR) and disallowed (outside CIDR) private addresses is denied."""
    cfg = FetchConfig(allow_private_hosts=("10.1.0.0/16",))
    # 10.1.2.3 is inside the allowed CIDR, but 172.16.0.1 is not.
    # The host resolves to both, so it's denied (all addresses must pass).
    with pytest.raises(BlockedAddressError):
        validate_url("http://internal.corp/x", cfg,
                     resolve=_resolver({"internal.corp": ["10.1.2.3", "172.16.0.1"]}))


# Tests for guarded_fetch

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _public(*hosts):
    return lambda host: ["93.184.216.34"] if host in hosts else ["127.0.0.1"]


def test_guarded_fetch_returns_a_direct_response():
    def handler(request):
        return httpx.Response(200, text="hello")

    with _client(handler) as c:
        resp = guarded_fetch("https://example.com/x", FetchConfig(), client=c, max_bytes=10_000,
                           resolve=_public("example.com"))
    assert resp.status_code == 200
    assert resp.body.decode() == "hello"


def test_guarded_fetch_follows_a_public_redirect():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "https://example.com/end"})
        return httpx.Response(200, text="arrived")

    with _client(handler) as c:
        resp = guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                           resolve=_public("example.com"))
    assert resp.body.decode() == "arrived"


def test_guarded_fetch_blocks_a_redirect_into_loopback():
    """The attack this exists to stop: a public URL 302-ing to an internal address."""
    def handler(request):
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8080/admin"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError, match="127.0.0.1"):
            guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


def test_guarded_fetch_blocks_a_redirect_to_the_metadata_ip():
    def handler(request):
        return httpx.Response(302,
                              headers={"Location": "http://169.254.169.254/latest/meta-data/"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError):
            guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


def test_guarded_fetch_resolves_a_relative_redirect_before_validating():
    """A relative Location must be joined against the current URL, or the host check
    would run against an empty host and the hop would escape validation."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/next"})
        return httpx.Response(200, text="ok")

    with _client(handler) as c:
        resp = guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                           resolve=_public("example.com"))
    assert resp.body.decode() == "ok"
    assert seen == ["https://example.com/start", "https://example.com/next"]


def test_guarded_fetch_enforces_max_redirects():
    def handler(request):
        return httpx.Response(302, headers={"Location": "https://example.com/loop"})

    with _client(handler) as c:
        with pytest.raises(httpx.HTTPError, match="too many redirects"):
            guarded_fetch("https://example.com/loop", FetchConfig(max_redirects=3), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


def test_guarded_fetch_blocks_the_initial_url_before_any_request():
    called = []

    def handler(request):
        called.append(str(request.url))
        return httpx.Response(200)

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError):
            guarded_fetch("http://10.0.0.1/x", FetchConfig(), client=c, max_bytes=10_000,
                          resolve=_public())
    assert called == []          # never left the process


def test_guarded_fetch_validates_hops_even_if_client_follows_redirects():
    """The guarantee must not depend on how the caller built their client.

    A client with follow_redirects=True would otherwise let httpx walk the chain
    internally, bypassing per-hop validation entirely while still appearing to work.
    """
    def handler(request):
        if "127.0.0.1" in str(request.url):
            return httpx.Response(200, text="internal data")
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8080/admin"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as c:
        with pytest.raises(BlockedAddressError, match="127.0.0.1"):
            guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


def test_guarded_fetch_enforces_max_redirects_with_exact_count():
    """Verify the redirect budget is exactly max_redirects (initial + redirects, not off-by-one)."""
    requests_made = []

    def handler(request):
        requests_made.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://example.com/loop"})

    with _client(handler) as c:
        with pytest.raises(httpx.HTTPError, match="too many redirects"):
            guarded_fetch("https://example.com/loop", FetchConfig(max_redirects=3), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))
    # Expect exactly 4 requests: initial + 3 allowed redirects
    assert len(requests_made) == 4


@pytest.mark.parametrize("status_code", [303, 307, 308])
def test_guarded_fetch_blocks_non_302_redirect_into_loopback(status_code):
    """Verify all redirect codes (not just 302) trigger hop validation."""
    def handler(request):
        return httpx.Response(status_code, headers={"Location": "http://127.0.0.1:8080/admin"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError, match="127.0.0.1"):
            guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


def test_guarded_fetch_blocks_redirect_to_public_metadata_address():
    """Test that 302 to a public metadata address (Azure wireserver) is blocked."""
    def handler(request):
        return httpx.Response(302,
                              headers={"Location": "http://168.63.129.16/metadata"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError, match="blocked cloud metadata endpoint"):
            guarded_fetch("https://example.com/start", FetchConfig(), client=c, max_bytes=10_000,
                        resolve=_public("example.com"))


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
                            max_bytes=2500, resolve=_public("example.com"))
    assert got.truncated and len(got.body) == 2500
    assert len(served) < 1000                       # the rest was never pulled


def test_guarded_fetch_enforces_a_total_deadline():
    ticks = iter(range(0, 10_000, 50))              # each clock() call advances 50 s

    def handler(request):
        return httpx.Response(200, content=(b"x" for _ in range(100)))

    with _client(handler) as c:
        with pytest.raises(FetchDeadlineExceeded):
            guarded_fetch("https://example.com/slow", FetchConfig(total_timeout_s=120),
                          client=c, max_bytes=10_000, resolve=_public("example.com"),
                          clock=lambda: next(ticks))


def test_guarded_fetch_rejects_a_redirect_to_a_disallowed_scheme():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://example.org/plain"})

    with _client(handler) as c:
        with pytest.raises(BlockedAddressError, match="scheme"):
            guarded_fetch("https://example.com/start", FetchConfig(allowed_schemes=("https",)),
                          client=c, max_bytes=10_000,
                          resolve=_public("example.com", "example.org"))


def test_guarded_fetch_reports_encoding_and_final_url():
    def handler(request):
        return httpx.Response(200, content="é".encode("latin-1"),
                              headers={"content-type": "text/plain; charset=latin-1"})

    with _client(handler) as c:
        got = guarded_fetch("https://example.com/x", FetchConfig(), client=c,
                            max_bytes=100, resolve=_public("example.com"))
    assert got.encoding == "latin-1" and got.url == "https://example.com/x"
    assert got.body.decode(got.encoding) == "é" and not got.truncated and not got.is_error


def _assert_drip_hits_the_deadline(preamble: bytes, drip):
    """Serve ``preamble`` then ``drip(n)`` every 0.2 s forever on loopback; guarded_fetch
    must raise FetchDeadlineExceeded well within 3 s. The server is always torn down."""
    import socket
    import threading
    import time

    stop = threading.Event()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(0.1)
    port = srv.getsockname()[1]
    conns = []

    def serve():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conns.append(conn)
            try:
                conn.sendall(preamble)
                n = 0
                while not stop.is_set():
                    conn.sendall(drip(n))
                    n += 1
                    stop.wait(0.2)
            except OSError:
                pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    cfg = FetchConfig(total_timeout_s=1, timeout_s=0.5, allow_private_hosts=("127.0.0.1",))
    started = time.monotonic()
    try:
        with httpx.Client(follow_redirects=False) as c:
            with pytest.raises(FetchDeadlineExceeded):
                guarded_fetch(f"http://127.0.0.1:{port}/drip", cfg, client=c,
                              max_bytes=10_000)
        assert time.monotonic() - started < 3
    finally:
        stop.set()
        t.join(timeout=5)
        for conn in conns:
            conn.close()
        srv.close()


def test_guarded_fetch_deadline_bounds_a_header_drip():
    """The client's timeout bounds each read, so a server dripping header lines forever
    never trips it; the total deadline must cut the request off during the header phase."""
    _assert_drip_hits_the_deadline(b"HTTP/1.1 200 OK\r\n", lambda n: b"X-Drip-%d: x\r\n" % n)


def test_guarded_fetch_deadline_mid_body_is_never_a_complete_response():
    """A body delimited by connection close: the watchdog's socket shutdown reads as a
    clean EOF, which must not be reported as a complete (untruncated) response."""
    _assert_drip_hits_the_deadline(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
                                   lambda n: b"a")
