"""build_tools: wrap the Session-bound tool impls as agent-ready callables.

Each wrapper has a clean model-facing signature (no ``session`` parameter), an
accurate ``__name__``, and a docstring that MAF turns into the tool description.
"""

from __future__ import annotations

from ..session import Session
# Import the impl functions directly. NB: importing the submodules with `from . import
# search` would get the *function* `search` (the package __init__ re-exports it, shadowing
# the submodule of the same name) -- so always alias the functions, not the modules.
from .code import run_python as _run_python
from .deliver import publish_file as _publish_file, publish_handle as _publish_handle
from .documents import read_document as _read_document
from .fetch import fetch_url as _fetch_url
from .files import list_files as _list_files, read_file as _read_file, write_file as _write_file
from .inspect import inspect_handle as _inspect_handle
from .search import search as _search
from .web import web_extract as _web_extract, web_search as _web_search


def build_tools(session: Session) -> list:
    """Return the agent's tool callables, each bound to ``session``."""

    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read up to `limit` lines starting at line `offset` from a file in the workspace."""
        return _read_file(session, path, offset, limit)

    def write_file(path: str, content: str) -> str:
        """Write `content` to a file in the workspace (creates parent directories)."""
        return _write_file(session, path, content)

    def list_files(path: str = ".") -> list[str]:
        """List files under `path` (a file or folder) in the workspace, recursively."""
        return _list_files(session, path)

    def search(pattern: str, path: str = ".", glob: str | None = None,
               ignore_case: bool = False, max_matches: int = 100) -> list[dict]:
        """Search files under `path` for the regex `pattern`. Returns file/line/col/text hits."""
        return _search(session, pattern, path, glob, ignore_case, max_matches)

    def fetch_url(url: str, max_bytes: int | None = None) -> dict:
        """Fetch `url` and store its body as a handle; returns the handle summary."""
        return _fetch_url(session, url, max_bytes)

    def run_python(code: str | None = None, path: str | None = None,
                   args: list[str] | None = None) -> dict:
        """Run Python in the sandbox. Give `code` (inline) or `path` (a script file). Scripts
        may use load(id)/save(id, obj)/emit(obj), and publish(path) when file delivery is
        enabled. Returns stdout/result/error/new_handles/published."""
        return _run_python(session, code, path, args)

    def inspect_handle(handle_id: str, rows: int = 20, stats: bool = False) -> dict:
        """Deeper look at a handle: up to `rows` (max 100) preview lines or head rows; for a
        dataframe, stats=True adds per-column min/max/null_count."""
        return _inspect_handle(session, handle_id, rows, stats)

    def web_search(query: str, max_results: int = 5) -> dict:
        """Search the web. Returns an answer + ranked results [{title,url,content,score}].
        Use the result urls with fetch_url or web_extract to read a page."""
        return _web_search(session, query, max_results)

    def web_extract(url: str) -> dict:
        """Fetch a URL's clean content via the search provider; returns a markdown handle."""
        return _web_extract(session, url)

    def read_document(source: str) -> dict:
        """Convert a document (PDF/Office/spreadsheet) at a workspace path or http(s) URL to a
        clean markdown handle with tables preserved; returns the handle summary."""
        return _read_document(session, source)

    def publish_file(path: str, name: str | None = None,
                     description: str | None = None) -> dict:
        """Deliver a finished file in the workspace (e.g. under outputs/) to the user. Returns
        its name/size/sha256 and the host's reply; publish only final files the user asked for."""
        return _publish_file(session, path, name, description)

    def publish_handle(handle_id: str, format: str | None = None, name: str | None = None,
                       description: str | None = None) -> dict:
        """Deliver a handle to the user as a file. Dataframes convert to `format`
        ('csv' | 'xlsx' | 'parquet' | 'json', default csv); other handles are delivered as-is."""
        return _publish_handle(session, handle_id, format, name, description)

    return [read_file, write_file, list_files, search, fetch_url, run_python, inspect_handle,
            web_search, web_extract, read_document, publish_file, publish_handle]
