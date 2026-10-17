"""Web research tools backed by Tavily (search + extract).

Failures return structured error dicts (never raise into the agent loop), matching the
fetch_url convention. A missing TAVILY_API_KEY returns a clear, actionable message.
"""

from __future__ import annotations

import httpx

from ..session import Session
from ..status import report_progress

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_SNIPPET_CHARS = 500


def _auth_headers(api_key: str) -> dict[str, str]:
    # Tavily accepts both the Authorization: Bearer header (documented standard) and an
    # api_key body field; we send the header for forward-compatibility and keep the body
    # field for older deployments.
    return {"Authorization": f"Bearer {api_key}"}


def web_search(session: Session, query: str, max_results: int = 5,
               client: httpx.Client | None = None) -> dict:
    """Search the web. Returns {"answer": str|None, "results": [{title, url, content, score}]}.
    `content` is a clipped extracted snippet. Use the urls with web_extract/fetch_url for more."""
    cfg = session.config.search
    if not cfg.api_key:
        return {"error": "web search unavailable: set TAVILY_API_KEY in .env"}
    report_progress(f"searching: {query}", tool="web_search")
    owns = client is None
    client = client or httpx.Client(timeout=cfg.timeout_s)
    try:
        try:
            resp = client.post(_SEARCH_URL, headers=_auth_headers(cfg.api_key), json={
                "api_key": cfg.api_key, "query": query,
                "max_results": max_results, "include_answer": True,
                "search_depth": "basic",
            })
        except httpx.HTTPError as e:
            return {"error": f"search request failed: {e}"}
        if resp.is_error:
            return {"error": f"search HTTP {resp.status_code}", "status": resp.status_code}
        data = resp.json()
        results = [{
            "title": r.get("title"),
            "url": r.get("url"),
            "content": (r.get("content") or "")[:_SNIPPET_CHARS],
            "score": r.get("score"),
        } for r in data.get("results", [])]
        return {"answer": data.get("answer"), "results": results}
    finally:
        if owns:
            client.close()


def web_extract(session: Session, url: str, client: httpx.Client | None = None) -> dict:
    """Fetch a URL's clean content via Tavily /extract, store it as a markdown handle, and
    return the handle summary. Returns a structured error on failure."""
    cfg = session.config.search
    if not cfg.api_key:
        return {"error": "web extract unavailable: set TAVILY_API_KEY in .env"}
    report_progress(f"extracting {url}", tool="web_extract")
    owns = client is None
    client = client or httpx.Client(timeout=cfg.timeout_s)
    try:
        try:
            resp = client.post(_EXTRACT_URL, headers=_auth_headers(cfg.api_key),
                               json={"api_key": cfg.api_key, "urls": [url]})
        except httpx.HTTPError as e:
            return {"error": f"extract request failed: {e}", "url": url}
        if resp.is_error:
            return {"error": f"extract HTTP {resp.status_code}", "status": resp.status_code, "url": url}
        results = resp.json().get("results", [])
        if not results:
            return {"error": "no content extracted", "url": url}
        content = results[0].get("raw_content") or results[0].get("content") or ""
        handle = session.store.put(content, source=f"web_extract({url})", kind="text")
        return handle.summary()
    finally:
        if owns:
            client.close()
