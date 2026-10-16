"""fetch_url: pull a link, spill its body to a typed handle, return the handle summary.

Network/HTTP failures are returned as structured error dicts (not raised) so the agent
can adapt -- e.g. follow a different link or report the problem -- rather than dead-ending
on a "function failed". Redirects are followed and a browser User-Agent is sent so
ordinary anti-bot pages don't 403 a bare client.
"""

from __future__ import annotations

import json
import mimetypes
import os
from urllib.parse import urlparse

import httpx

from ..config import FetchConfig
from ..egress import BlockedAddressError, guarded_fetch
from ..session import Session
from ..status import report_progress

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TEXTUAL = ("json", "xml", "html", "csv", "javascript")


def _default_client(cfg: FetchConfig) -> httpx.Client:
    # follow_redirects=False: guarded_fetch follows hops itself so each one is re-validated
    # against egress policy. Letting httpx follow them would skip that check.
    return httpx.Client(
        timeout=cfg.timeout_s,
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT},
    )


def _is_binary(content_type: str, body: bytes) -> bool:
    """True if the payload should be stored as raw bytes (xls/pdf/image/zip/...) rather than
    decoded as text. Textual content-types pass through; otherwise sniff a utf-8 decode."""
    ct = content_type.lower()
    if ct.startswith("text/") or any(k in ct for k in _TEXTUAL):
        return False
    try:
        body[:8192].decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def _ext_for(url: str, content_type: str) -> str:
    """Best-effort file extension for a binary handle: prefer the URL suffix, else the
    content-type's registered extension, else ``.bin``."""
    suffix = os.path.splitext(urlparse(url).path)[1]
    if suffix and len(suffix) <= 6:
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    return guessed or ".bin"


def fetch_url(session: Session, url: str, max_bytes: int | None = None,
              raw: bool = False, client: httpx.Client | None = None) -> dict:
    """Fetch ``url`` and store its body as a handle (json if JSON content-type, else text).

    HTML responses are converted to clean markdown via trafilatura (stripping nav, footer,
    ads, and other boilerplate) unless ``raw=True`` is passed, in which case the raw HTML
    text is stored unchanged.

    Returns the handle summary on success, or ``{"error", "status", "url"}`` on an HTTP
    error / network failure. Follows redirects, re-validating each hop against egress
    policy, and sends a browser User-Agent. Enforces the session's allowed schemes and
    byte cap. ``client`` is injectable for testing.
    """
    cfg = session.config.fetch
    # The per-call limit is model-supplied; the ceiling is the host's.
    limit = cfg.max_bytes if max_bytes is None else max(1, min(int(max_bytes),
                                                               cfg.max_bytes_ceiling))

    scheme = urlparse(url).scheme
    if scheme not in cfg.allowed_schemes:
        # Structured, like every other failure here and like read_document's handling of
        # exactly this case. Raising made one of this tool's failure modes arrive as a
        # "function failed" the model could not read or adapt to.
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

    try:
        text = body.decode(got.encoding, errors="replace")
    except (LookupError, UnicodeError):
        # The charset is server-chosen: unknown to Python (LookupError), or a real codec
        # that rejects errors="replace" (e.g. idna raises UnicodeError). Fall back to utf-8.
        text = body.decode("utf-8", errors="replace")
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
