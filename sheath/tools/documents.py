"""read_document: convert a PDF/Office/spreadsheet to clean markdown (with tables) via Docling.

The source may be a workspace-relative path (resolved + jailed by safe_path) or an http(s)
URL (downloaded through the egress guard into the harness-private scratch directory
``.sheath/tmp/`` first, so Docling only ever receives a local path and never makes its own
network requests). A workspace file is likewise snapshotted into scratch under the session
lock, so sandboxed code cannot swap it while it is being read. The markdown is stored as a text handle
and only the summary is returned, keeping large document text out of the model's context.

Following the fetch_url/web convention, every failure -- a path escaping the root, an unknown
scheme, an unsupported/corrupt file, or Docling not being installed -- is returned as a
structured {"error", "source"} dict rather than raised into the agent loop. The `convert`
seam is injectable so unit tests never need real Docling (heavy, downloads models).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

from .. import workspace
from ..egress import BlockedAddressError, guarded_fetch
from ..paths import PathEscapesRootError, safe_path
from ..sandbox import SandboxQuarantinedError
from ..session import Session
from ..status import report_progress
from .files import _open_checked

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _docling_convert(source: str, ocr: bool = False) -> str:
    """Convert a local path or URL to markdown (tables preserved) via Docling. Lazy import:
    Docling is an optional, heavy dependency, so it is only imported when actually used.

    OCR defaults off: born-digital documents get their tables/structure from the layout and
    TableFormer models, so OCR only adds latency and extra model downloads. Pass ``ocr=True``
    for scanned/image documents.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline = PdfPipelineOptions(do_ocr=ocr)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    return converter.convert(source).document.export_to_markdown()


_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def _safe_suffix(url: str) -> str:
    """A short alphanumeric extension, so Docling can infer the format.

    The URL is model-controlled, so the suffix is allowlisted rather than sanitised:
    an overlong one makes mkstemp raise OSError, which would escape this tool's
    structured-error contract. Path.suffix cannot contain a separator, so traversal
    is not a concern here -- length and character class are.
    """
    suffix = Path(urlparse(url).path).suffix
    return suffix if _SAFE_SUFFIX.match(suffix) else ".bin"


def _scratch_file(session: Session, suffix: str) -> Path:
    """A fresh file in the parent-only scratch directory (never in the model's work area)."""
    fd, abspath = tempfile.mkstemp(prefix="doc_", suffix=suffix,
                                   dir=workspace.scratch_dir(session.root))
    os.close(fd)
    return Path(abspath)


def _download(session: Session, url: str) -> Path:
    """Fetch ``url`` through the egress guard into private scratch; return the local path.

    The filename keeps the URL's extension so Docling can infer the format. A download
    that hits the byte cap is refused: converting a partial document yields garbage.
    """
    cfg = session.config.fetch
    client = httpx.Client(timeout=cfg.timeout_s, follow_redirects=False,
                          headers={"User-Agent": _USER_AGENT})
    try:
        got = guarded_fetch(url, cfg, client=client, max_bytes=cfg.max_bytes)
    finally:
        client.close()
    if got.is_error:  # 4xx/5xx -> reported as a download failure by the caller
        raise httpx.HTTPError(f"HTTP {got.status_code}")
    if got.truncated:
        raise httpx.HTTPError(f"document is larger than FetchConfig.max_bytes ({cfg.max_bytes})")
    dest = _scratch_file(session, _safe_suffix(url))
    dest.write_bytes(got.body)
    return dest


def _snapshot(session: Session, source: str) -> Path:
    """Copy a workspace file into private scratch under the session lock.

    The model's work area is writable by sandboxed code, so the file is opened once
    (never following a final symlink, never blocking on a FIFO), checked to be a regular
    file on the descriptor itself, and copied from that descriptor while no sandboxed code
    can run; conversion (slow) then happens on the private copy, outside the lock.
    """
    limit = session.config.max_input_bytes
    with session.file_io():
        target = safe_path(session.root, source)
        fd = _open_checked(target, os.O_RDONLY, source)
        with open(fd, "rb") as src:
            size = os.fstat(src.fileno()).st_size   # the opened file, not the name
            if size > limit:
                raise ValueError(f"document too large ({size} bytes > {limit})")
            suffix = target.suffix if _SAFE_SUFFIX.match(target.suffix) else ".bin"
            dest = _scratch_file(session, suffix)
            try:
                with dest.open("wb") as out:
                    copied = 0
                    while chunk := src.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > limit:
                            raise ValueError(f"document too large (more than {limit} bytes)")
                        out.write(chunk)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
    return dest


def read_document(session: Session, source: str,
                  convert: Callable[[str], str] | None = None) -> dict:
    """Convert ``source`` (a workspace path or http(s) URL) to a clean-markdown handle.

    Returns the handle summary on success, or ``{"error", "source"}`` on any failure.
    ``convert`` is an injectable converter (defaults to Docling, honoring
    ``config.documents.ocr``) for testing.
    """
    cfg = session.config.fetch
    scratch: Path | None = None
    scheme = urlparse(source).scheme
    # The session's allowed schemes, not a hard-coded pair: narrowing
    # FetchConfig.allowed_schemes to ("https",) must bind this tool as well as fetch_url.
    if scheme in cfg.allowed_schemes:
        # Docling fetches URLs itself and follows its own redirects, which egress policy
        # cannot see. Download through the guard instead and convert the local file.
        try:
            scratch = _download(session, source)
        except BlockedAddressError as e:
            return {"error": f"blocked by egress policy: {e}", "source": source}
        except httpx.HTTPError as e:
            return {"error": f"could not download {source!r}: {e}", "source": source}
        except OSError as e:
            return {"error": f"could not store {source!r}: {e}", "source": source}
    elif scheme == "":
        try:
            scratch = _snapshot(session, source)
        except PathEscapesRootError:
            return {"error": f"path escapes the workspace root: {source!r}", "source": source}
        except (OSError, ValueError, SandboxQuarantinedError) as e:
            return {"error": f"could not read {source!r}: {e}", "source": source}
    else:
        return {"error": f"unsupported source scheme {scheme!r}; pass a workspace path or a "
                         f"URL with one of {cfg.allowed_schemes}", "source": source}
    target = str(scratch)

    if convert is None:
        ocr = session.config.documents.ocr
        def convert(src: str) -> str:  # default converter honors the session's OCR setting
            return _docling_convert(src, ocr=ocr)

    report_progress(f"converting {source} via Docling", tool="read_document")
    try:
        markdown = convert(target)
    except ImportError:
        return {"error": "document ingestion unavailable: install the 'docling' extra "
                         "(e.g. `uv sync --extra docling`) to enable Docling", "source": source}
    except Exception as e:  # noqa: BLE001 - unsupported/corrupt file etc. -> structured error
        return {"error": f"could not read document: {e}", "source": source}
    finally:
        # Scratch only: the markdown lives in a handle, and leaving these behind would give
        # a model looping on read_document an unbounded disk-growth primitive.
        scratch.unlink(missing_ok=True)

    handle = session.store.put(markdown, source=f"read_document({source})", kind="text")
    return handle.summary()


def prefetch_models(downloader: Callable[..., object] | None = None, *, ocr: bool = False) -> None:
    """Pre-download Docling's models so the first ``read_document`` call doesn't pay for it.

    Fetches the layout + TableFormer (and related) models by default; OCR models are pulled
    only when ``ocr=True``, matching the OCR-off conversion default. ``downloader`` is
    injectable for testing; by default it lazily imports Docling's model downloader.
    """
    if downloader is None:
        from docling.utils.model_downloader import download_models as downloader

    downloader(progress=True, with_rapidocr=ocr)


def _prefetch_main() -> None:
    """Console-script entry (``sheath-prefetch-docling``): warm the Docling model cache."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sheath-prefetch-docling",
        description="Pre-download Docling models so the first read_document call is fast.",
    )
    parser.add_argument("--ocr", action="store_true",
                        help="also fetch OCR models (needed only for scanned/image documents)")
    args = parser.parse_args()
    prefetch_models(ocr=args.ocr)
    print("Docling models prefetched.")
