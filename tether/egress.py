"""Egress policy: the single place that decides whether a URL may be fetched.

The model chooses the URLs this project fetches, and fetched content is itself untrusted,
so egress is a model-controlled capability. This module denies the internal address space
and cloud instance-metadata endpoints by default. Metadata endpoints are checked
unconditionally and cannot be overridden by the allowlist.

When following redirects, guarded_fetch re-validates each hop (address and scheme) against
egress policy, because httpx's redirect-following happens without consulting policy. Without
this, a public URL could 302 straight into the internal network or metadata endpoints. It
also streams the body with a byte cap and bounds the whole fetch with a total deadline.

Known residual risk: validation resolves the hostname and then hands the URL to httpx,
which resolves it again -- a DNS entry that changes in between (rebinding) is not caught.
Closing that needs connect-to-validated-IP with a Host override; see the spec's out-of-scope
section.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from .config import FetchConfig

# RFC 6598 carrier-grade NAT. Verified: no ipaddress flag reports this range private,
# reserved, or otherwise special, yet it is routinely routable inside cloud and carrier
# networks. Every other special-use range is already covered by the flag checks below.
_EXTRA_DENIED_NETS = (ipaddress.ip_network("100.64.0.0/10"),)

# Cloud instance-metadata endpoints. Never a legitimate data source, and reaching one
# usually means credential theft. These are denied unconditionally, regardless of allowlist
# configuration. Allowlisting a hostname vouches for the name, not for whatever it resolves
# to later.
_METADATA_ADDRS = frozenset(ipaddress.ip_address(a) for a in (
    "169.254.169.254",   # AWS / GCP / Azure IMDS, OpenStack, and most others
    "169.254.170.2",     # AWS ECS task credentials
    "168.63.129.16",     # Azure wireserver -- a PUBLIC address, denied by nothing else
    "100.100.100.200",   # Alibaba Cloud (inside the CGNAT range)
    "192.0.0.192",       # Oracle Cloud
    "fd00:ec2::254",     # AWS IMDS over IPv6
))


class BlockedAddressError(ValueError):
    """Raised when a URL's host resolves to an address egress policy forbids."""


def _resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _is_metadata(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Cloud instance-metadata endpoints. Never a legitimate data source, and reaching one
    usually means credential theft -- so these are denied unconditionally, ahead of the
    allowlist. Allowlisting a hostname vouches for the name, not for whatever it may
    resolve to later.
    """
    mapped = getattr(addr, "ipv4_mapped", None)
    return (mapped or addr) in _METADATA_ADDRS


def _is_denied(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if any(addr in net for net in _EXTRA_DENIED_NETS):
        return True
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def _allowed_by_config(host: str, addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
                       cfg: FetchConfig) -> bool:
    for entry in cfg.allow_private_hosts:
        if entry.lower() == host.lower():
            return True
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue       # a hostname entry, already compared above
    return False


def validate_url(url: str, cfg: FetchConfig, *,
                 resolve: Callable[[str], list[str]] | None = None) -> None:
    """Raise BlockedAddressError unless every address ``url``'s host resolves to is allowed.

    *Every* address must pass: a host answering with both a public and an internal address
    is denied, because the attacker -- not us -- effectively picks which one is connected to.
    ``resolve`` is injectable so tests never perform DNS.
    """
    host = urlparse(url).hostname
    if not host:
        raise BlockedAddressError(f"no host in url: {url!r}")

    try:                                  # a literal IP needs no resolution
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        resolver = resolve or _resolve
        try:
            addrs = resolver(host)
        except (OSError, UnicodeError) as e:
            raise BlockedAddressError(f"could not resolve host {host!r}: {e}") from e
        if not addrs:
            raise BlockedAddressError(f"could not resolve host {host!r}: no addresses")

    for raw in addrs:
        addr = ipaddress.ip_address(raw)
        # Metadata endpoints are checked unconditionally, before allowlist.
        if _is_metadata(addr):
            raise BlockedAddressError(
                f"blocked cloud metadata endpoint {raw} for host {host!r}")
        if _is_denied(addr) and not _allowed_by_config(host, addr, cfg):
            raise BlockedAddressError(
                f"blocked internal address {raw} for host {host!r}; add the host or its "
                f"CIDR to FetchConfig.allow_private_hosts to permit it")


_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


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
    stops at ``max_bytes`` (the response is then closed, so the remainder is never pulled).

    The whole operation is bounded by ``cfg.total_timeout_s`` -- the client's own timeout
    only bounds each read, which a slow drip (of headers or body) never trips. Two
    mechanisms enforce it: ``clock`` is checked before each hop and after each body chunk,
    and a watchdog timer shuts down the socket of every connection this call opened, so a
    read blocked in the header or body phase returns at once; the resulting transport
    error is reported as ``FetchDeadlineExceeded``. Sockets are found through httpcore's
    ``trace`` request extension, so the client itself is left open and usable (closing an
    fd under a blocked reader does not wake it, and its number could be reused).

    Residual, by design: DNS resolution inside ``validate_url`` cannot be interrupted, a
    TCP/TLS connect is bounded only by the client's per-phase timeout (the watchdog shuts
    the socket as soon as the connect completes), and a keep-alive connection the client
    opened *before* this call is not traced -- both production callers use a fresh client.
    The caller owns the client's lifecycle.
    """
    deadline = clock() + cfg.total_timeout_s
    expired = threading.Event()
    streams: list = []
    lock = threading.Lock()

    def shut(stream) -> None:
        try:
            stream.get_extra_info("socket").shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass                  # already closed, or a transport without a real socket

    def trace(event: str, info: dict) -> None:
        if event in ("connection.connect_tcp.complete", "connection.start_tls.complete"):
            stream = info.get("return_value")
            with lock:
                streams.append(stream)
                late = expired.is_set()
            if late:              # connected after the deadline: cut it off immediately
                shut(stream)

    def expire() -> None:
        with lock:
            expired.set()
            pending = list(streams)
        for stream in pending:
            shut(stream)

    def deadline_error() -> FetchDeadlineExceeded:
        return FetchDeadlineExceeded(
            f"fetch exceeded the {cfg.total_timeout_s:.0f}s total deadline: {url!r}")

    def check_deadline() -> None:
        if expired.is_set() or clock() > deadline:
            raise deadline_error()

    watchdog = threading.Timer(cfg.total_timeout_s, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        return _fetch_hops(url, cfg, client=client, max_bytes=max_bytes, resolve=resolve,
                           check_deadline=check_deadline, trace=trace)
    except FetchDeadlineExceeded:
        raise
    except (httpx.HTTPError, OSError) as e:
        # A socket shut down by the watchdog surfaces as an ordinary transport error.
        if expired.is_set():
            raise deadline_error() from e
        raise
    finally:
        watchdog.cancel()


def _fetch_hops(url: str, cfg: FetchConfig, *, client: httpx.Client, max_bytes: int,
                resolve: Callable[[str], list[str]] | None,
                check_deadline: Callable[[], None],
                trace: Callable[[str, dict], None]) -> FetchedBody:
    current = url
    for _ in range(cfg.max_redirects + 1):
        check_deadline()
        scheme = urlparse(current).scheme
        if scheme not in cfg.allowed_schemes:
            raise BlockedAddressError(
                f"scheme {scheme!r} not allowed for {current!r} (allowed: {cfg.allowed_schemes})")
        validate_url(current, cfg, resolve=resolve)
        with client.stream("GET", current, follow_redirects=False,
                           extensions={"trace": trace}) as resp:
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
