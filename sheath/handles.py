"""Typed handles: large data lives on disk; only a lightweight summary enters context."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import workspace
from .paths import PathEscapesRootError, safe_path

_PREVIEW_CHARS = 800
_PREVIEW_ROWS = 5
_MAX_SCHEMA_COLS = 200
_MAX_COL_NAME_CHARS = 128
_MAX_SOURCE_CHARS = 200
_DIGEST_BYTES = 64 * 1024
_MAX_LEGACY_MANIFEST_BYTES = 8 * 1024 * 1024   # migration reads the old, sandbox-writable file
# Arrow decodes a whole column chunk of the first row group to yield even one small batch, so
# a preview is skipped when the footer says that group is larger than this.
_MAX_PREVIEW_GROUP_BYTES = 16 * 1024 * 1024
# Uploaded-input previews: raw first lines, for text-like extensions only. Never parsed.
_TEXT_PREVIEW_EXTS = {".csv", ".tsv", ".txt", ".json", ".md"}
_PREVIEW_LINES = 5
_PREVIEW_READ_BYTES = 4096

# Ids that sandboxed code may name a handle with. One path segment, no leading dot, so the
# id can be used directly as a filename stem under handles/.
HANDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_KIND_EXT = {"dataframe": ".parquet", "json": ".json", "text": ".txt"}


_DIGEST_TAG = "v2:"


def _digest_file(path: Path) -> str:
    """Integrity digest for a handle's bytes: ``"v2:"`` + sha256 over the file's length, its
    first ``_DIGEST_BYTES`` (64 KiB) and its last ``_DIGEST_BYTES``.

    **Coverage, exactly.** A file of 128 KiB or less is hashed *in its entirety*: an edit at
    any offset is detected. A larger file is covered by its length, its first 64 KiB and its
    last 64 KiB. A length-preserving edit to the **middle** of a file larger than 128 KiB
    (the bytes between offset 64 KiB and ``size - 64 KiB``) is **not detected**. This is not
    a whole-file integrity check.

    The tail is covered because a parquet footer -- the schema and row count the summary
    reports -- lives there, and the head alone left it unprotected.

    **Why not a whole-file hash.** The describers read only a bounded window so a
    multi-gigabyte handle can be described cheaply; hashing the whole file on every ``get()``
    would throw that away. The residual above is the stated price.

    **Boundary handling.** For ``size <= 2 * _DIGEST_BYTES`` the two windows would overlap
    or touch, so the file is read once, whole, and hashed once (no byte is hashed twice). For
    larger files the head and tail windows are disjoint, read with one seek on one open
    handle. The size is taken from that open handle, so the length hashed is the length read.

    **Format tag.** The ``v2:`` prefix distinguishes this from the earlier untagged
    (length + first 64 KiB) digests. A digest recorded in the old format therefore never
    equals a current one: it fails verification (a false failure) rather than passing
    (a false pass). Handle digests are re-derived on rehydrate, so this only affects a
    digest held across a code upgrade within one process.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        h.update(f"{size}:".encode())
        if size <= 2 * _DIGEST_BYTES:
            h.update(f.read())
        else:
            h.update(f.read(_DIGEST_BYTES))
            f.seek(size - _DIGEST_BYTES)
            h.update(f.read(_DIGEST_BYTES))
    return _DIGEST_TAG + h.hexdigest()


def first_group_too_large(pf: Any) -> int | None:
    """The footer's byte size of row group 0 if it exceeds ``_MAX_PREVIEW_GROUP_BYTES``,
    else ``None``. Callers must not decode a preview from a file for which this returns a size.

    The size is sandbox-authored footer metadata: a lying footer can defeat this guard.
    """
    if not pf.num_row_groups:
        return None
    size = int(pf.metadata.row_group(0).total_byte_size)
    return size if size > _MAX_PREVIEW_GROUP_BYTES else None


class HandleTamperedError(RuntimeError):
    """Raised when a handle's bytes changed after its metadata was derived.

    Not a recoverable condition: the summary the model is holding describes data that is no
    longer on disk, so continuing would feed it content nothing has vouched for.
    """

    def __init__(self, handle_id: str, path: str) -> None:
        super().__init__(
            f"handle {handle_id!r} ({path}) changed on disk after it was created: its "
            f"content digest no longer matches the one recorded when its metadata was "
            f"derived. The summary already in context describes different bytes. This is a "
            f"tamper signal, not a recoverable condition -- save a new handle instead of "
            f"overwriting an existing one."
        )
        self.handle_id = handle_id


class HandleIdReuseError(ValueError):
    """Raised when adopting an id that already exists. Handles are immutable.

    Subclasses ValueError so every existing `except ValueError` caller keeps working;
    ingestion catches it by type to report it rather than silently skipping.
    """

    def __init__(self, handle_id: str) -> None:
        super().__init__(f"handle id {handle_id!r} already exists; handles are immutable")
        self.handle_id = handle_id


@dataclass
class Handle:
    id: str
    kind: str  # "json" | "text" | "dataframe" | "binary"
    path: str  # POSIX path relative to the session root
    source: str
    bytes: int
    preview: str
    schema: dict[str, str] | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    input_id: str | None = None       # host-supplied id of an uploaded input
    content_type: str | None = None   # advisory type of an uploaded input
    description: str | None = None    # host-supplied description of an uploaded input
    # Integrity only, never shown to the model. ``None`` means nothing was recorded (a
    # Handle built by hand, or a record predating digests) -- get() then has nothing to
    # compare against and reads anyway.
    digest: str | None = None

    def summary(self) -> dict[str, Any]:
        """Context-facing view: drop None fields to keep it compact.

        The digest is dropped too: it is an internal integrity field, 64 hex characters of
        pure noise in model context, and it is re-derived from the file on rehydration
        rather than read back from the manifest.
        """
        return {k: v for k, v in asdict(self).items()
                if v is not None and k != "digest"}


class HandleStore:
    """Persists objects under ``<root>/handles`` and tracks them by id."""

    # json and text describe identically, so both kinds dispatch to the one shared body.
    _DESCRIBERS = {"dataframe": "_describe_dataframe", "json": "_describe_textual",
                   "text": "_describe_textual", "binary": "_describe_binary"}

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Creates handles/ and .sheath/ as real directories, or refuses the root if either
        # exists as a link: every write below lands inside one of them.
        workspace.ensure_layout(self.root)
        self.dir = self.root / "handles"
        self._state = workspace.state_dir(self.root)
        self._handles: dict[str, Handle] = {}
        self._counter = 0
        # Tool calls in one turn run on worker threads in parallel; every mutation of the
        # id counter, the handle map and the manifest happens under this lock.
        self._lock = threading.RLock()
        self._load_manifest()

    def _new_id(self) -> str:
        self._counter += 1
        return f"h{self._counter}"

    def _atomic(self, dst: Path, write: Callable[[Path], None]) -> None:
        """Write ``dst`` via a unique temp file in the same directory, then rename.

        ``os.replace`` swaps the directory entry, so a link planted at ``dst`` is replaced,
        never written through, and a reader never sees a half-written file.
        """
        fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp_")
        os.close(fd)
        try:
            write(Path(tmp))
            os.chmod(tmp, 0o644)
            os.replace(tmp, dst)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _detect_kind(obj: Any) -> str:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return "dataframe"
        if isinstance(obj, (dict, list)):
            return "json"
        if isinstance(obj, str):
            return "text"
        if isinstance(obj, (bytes, bytearray)):
            return "binary"
        raise TypeError(f"unsupported handle object type: {type(obj)!r}")

    def put(self, obj: Any, source: str, *, id: str | None = None,
            kind: str | None = None, ext: str | None = None) -> Handle:
        kind = kind or self._detect_kind(obj)
        with self._lock:
            hid = id or self._new_id()
            if id is not None:
                self._advance_counter(hid)  # keep auto-ids from colliding with an explicit id
            if kind == "dataframe":
                handle = self._write_dataframe(hid, obj, source)
            elif kind == "json":
                handle = self._write_json(hid, obj, source)
            elif kind == "text":
                handle = self._write_text(hid, obj, source)
            elif kind == "binary":
                handle = self._write_binary(hid, obj, source, ext)
            else:
                raise ValueError(f"unknown handle kind: {kind!r}")
            self._handles[hid] = handle
            self._save_manifest()
            return handle

    def _write_binary(self, hid: str, data: bytes, source: str, ext: str | None) -> Handle:
        # Store raw bytes intact so the file (xls/pdf/image/...) stays readable by pandas,
        # Docling, etc. The extension is preserved so libraries can infer the format.
        rel = f"handles/{hid}{ext or '.bin'}"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_bytes(bytes(data)))
        return Handle(id=hid, kind="binary", path=rel, source=source,
                      **self._describe("binary", path))

    def _write_dataframe(self, hid: str, df: Any, source: str) -> Handle:
        rel = f"handles/{hid}.parquet"
        path = self.root / rel
        self._atomic(path, lambda tmp: df.to_parquet(tmp))
        return Handle(id=hid, kind="dataframe", path=rel, source=source,
                      **self._describe("dataframe", path))

    def _write_json(self, hid: str, obj: Any, source: str) -> Handle:
        # ``default=str`` keeps non-JSON-native types (datetime, Decimal, ...) from
        # crashing serialization, but they round-trip back as strings via get().
        rel = f"handles/{hid}.json"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_text(json.dumps(obj, default=str),
                                                     encoding="utf-8"))
        return Handle(id=hid, kind="json", path=rel, source=source,
                      **self._describe("json", path))

    def _write_text(self, hid: str, obj: str, source: str) -> Handle:
        rel = f"handles/{hid}.txt"
        path = self.root / rel
        self._atomic(path, lambda tmp: tmp.write_text(obj, encoding="utf-8"))
        return Handle(id=hid, kind="text", path=rel, source=source,
                      **self._describe("text", path))

    def _describe(self, kind: str, path: Path) -> dict[str, Any]:
        """Derive every described field for ``kind`` from the bytes at ``path``.

        The single place metadata is produced: ``put``, ``adopt`` and manifest rehydration
        all route through here, so a handle's summary always describes the file, whoever
        wrote it. The integrity digest is taken here too, alongside the describer, so it is
        recorded at exactly the moment the metadata it protects is derived.
        """
        describer = self._DESCRIBERS.get(kind)
        if describer is None:
            raise ValueError(f"unknown handle kind: {kind!r}")
        described = getattr(self, describer)(path)
        described["digest"] = _digest_file(path)
        return described

    def _describe_dataframe(self, path: Path) -> dict[str, Any]:
        """Describe a parquet file without materializing it.

        Schema and row count come from the footer; the preview decodes only the first
        record batch of ``_PREVIEW_ROWS`` rows, never a whole row group (a row group in a
        file sandboxed code wrote can be arbitrarily large). The preview text and the
        schema listing are capped, because cell values and column names are untrusted and
        this summary goes into model context.

        Bounded: the preview text, the schema listing, and (via the footer-size guard
        ``first_group_too_large``) the memory spent on the preview -- a first row group over
        ``_MAX_PREVIEW_GROUP_BYTES`` is not decoded at all. Not bounded: footer parsing
        itself, and the footer's own claims -- sizes are sandbox-authored, so a footer that
        understates its row group defeats the guard (a disclosed residual).
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        n_rows = int(pf.metadata.num_rows)
        schema_df = pf.schema_arrow.empty_table().to_pandas()
        too_large = first_group_too_large(pf)
        batch = (None if too_large is not None
                 else next(pf.iter_batches(batch_size=_PREVIEW_ROWS), None))
        shown = 0
        preview = ""
        if batch is not None:
            head = batch.to_pandas()
            shown = len(head)
            preview = head.to_csv(index=False)
        elif too_large is not None:
            preview = (f"<preview omitted: first row group is {too_large} bytes; "
                       f"inspect it in run_python>")
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS] + "\n"
        # Caption the rows actually shown, not _PREVIEW_ROWS.
        if n_rows > shown:
            preview += f"... ({shown} of {n_rows} rows)"
        dtypes = list(schema_df.dtypes.items())
        return {
            "bytes": path.stat().st_size,
            "preview": preview,
            "schema": {str(c)[:_MAX_COL_NAME_CHARS]: str(t)
                       for c, t in dtypes[:_MAX_SCHEMA_COLS]},
            "n_rows": n_rows,
            "n_cols": int(len(dtypes)),
        }

    def _describe_textual(self, path: Path) -> dict[str, Any]:
        """Describe a json or text handle. One body serves both kinds: their summaries are
        identical, and two copies of this would drift the first time one format needed
        different preview handling.

        Reads only the preview window, never the whole file: ``adopt`` routes
        sandbox-authored files through here, and the container tier does not cap how large
        a file the child may write. ``f.read(n)`` on a text stream returns n *characters*
        with multibyte sequences handled, and ``st_size`` is the file's byte length -- which
        for the UTF-8 files ``put`` writes equals the encoded length of the text.
        """
        with path.open(encoding="utf-8") as f:
            preview = f.read(_PREVIEW_CHARS)
        return {"bytes": path.stat().st_size, "preview": preview}

    def _describe_binary(self, path: Path) -> dict[str, Any]:
        size = path.stat().st_size
        ext = path.suffix or ".bin"
        return {"bytes": size, "preview": f"<binary file, {size} bytes, {ext}>"}

    def _advance_counter(self, hid: str) -> None:
        """Keep the auto-id counter ahead of an externally-supplied ``h<N>`` id."""
        if hid.startswith("h") and hid[1:].isdigit():
            self._counter = max(self._counter, int(hid[1:]))

    def _register_record(self, record: dict[str, Any]) -> Handle:
        """Register a handle whose file already exists (manifest rehydration, or ``register``).

        **The record is untrusted.** The manifest lives at ``<root>/.sheath/manifest.json``,
        which the container tier mounts read-only, but the local tier gives sandboxed code
        no such barrier, and a manifest migrated from the pre-trust-zone location
        (``handles/_manifest.json``) was sandbox-writable. Only ``id``, ``kind``, ``path`` and ``source`` are taken from the record -- the
        things the writer alone knows. Every described field (``preview``, ``bytes``,
        ``schema``, ``n_rows``, ``n_cols``) is re-derived from the file by the same
        ``_describe`` that ``put`` and ``adopt`` use, so a forged manifest cannot make the
        model believe a false summary. The digest is re-derived too: the stored one offers
        nothing, since whoever could forge the metadata could forge the digest beside it.

        ``path`` goes through ``safe_path``, and the target must be a regular file -- a
        record pointing outside the root, at a missing file, or at a directory/FIFO/device
        is rejected here rather than hung on or read later.
        """
        try:
            hid, kind, path, source = (record["id"], record["kind"], record["path"],
                                       record.get("source", "unknown"))
        except (KeyError, TypeError) as e:  # contract boundary — give a useful message
            raise ValueError(f"invalid handle record {record!r}: {e}") from e
        if not (isinstance(hid, str) and isinstance(path, str)):
            raise ValueError(f"invalid handle record {record!r}: id and path must be strings")
        try:
            resolved = safe_path(self.root, path)
        except PathEscapesRootError as e:
            raise ValueError(f"handle record path escapes root: {record!r}") from e
        # safe_path already resolved symlinks, so compare the unresolved path to catch them.
        if not resolved.is_file() or (self.root / path).is_symlink():
            raise ValueError(f"handle record path is not a regular file: {record!r}")
        # ``input_id`` is the one record field that cannot be derived from the bytes: it
        # names *which* host upload this file is, and losing it on rehydration would break
        # add_input's idempotency across restarts. It is carried, not believed -- Session
        # re-checks a recorded input against the filesystem and against the host's bytes
        # before reusing it. ``content_type`` is re-derived from the filename and
        # ``description`` is dropped: both reach model context, and neither is recoverable
        # from a manifest sandboxed code can rewrite.
        input_id = record.get("input_id") if isinstance(record, dict) else None
        if not (isinstance(input_id, str) and input_id):
            input_id = None
        try:
            if input_id is not None:
                handle = Handle(id=hid, kind="binary", path=path, source=source,
                                input_id=input_id, **self._describe_input(resolved, None))
            else:
                handle = Handle(id=hid, kind=kind, path=path, source=source,
                                **self._describe(kind, resolved))
        except (OSError, TypeError) as e:  # unreadable file, or an unusable id/kind type
            raise ValueError(f"invalid handle record {record!r}: {e}") from e
        self._handles[handle.id] = handle
        self._advance_counter(handle.id)
        return handle

    def register(self, record: dict[str, Any]) -> Handle:
        """Register a handle whose file already exists; persists the manifest."""
        with self._lock:
            handle = self._register_record(record)
            self._save_manifest()
            return handle

    def adopt(self, *, id: str, kind: str, path: str, source: str) -> Handle:
        """Register a file written by sandboxed code, deriving its metadata here.

        The child reports only what it alone knows -- which file it wrote, and why. Every
        field that reaches model context (preview, bytes, schema, n_rows, n_cols) is
        computed from the bytes on disk, so a hostile child cannot describe its output
        falsely. Handles are immutable: adopting an id that already exists is refused, so
        the record for a handle the child did not create cannot be repointed.
        """
        with self._lock:
            if id in self._handles:
                raise HandleIdReuseError(id)
            describer = self._DESCRIBERS.get(kind)
            if describer is None:
                raise ValueError(f"unknown handle kind: {kind!r}")
            resolved = safe_path(self.root, path)  # raises PathEscapesRootError -> ValueError subclass
            # A directory, symlink, FIFO or device would hang or mislead the describer. safe_path
            # already resolved symlinks, so compare against the unresolved path to catch them.
            if not resolved.is_file() or (self.root / path).is_symlink():
                raise ValueError(f"handle record path is not a regular file: {path!r}")
            described = self._describe(kind, resolved)
            handle = Handle(id=id, kind=kind, path=path, source=source, **described)
            self._handles[id] = handle
            self._advance_counter(id)
            self._save_manifest()
            return handle

    def adopt_file(self, *, id: str, kind: str, src: Path, source: str) -> Handle:
        """Move a payload sandboxed code wrote in its run directory into ``handles/`` and
        register it, deriving every described field here.

        The sandbox names only the id, the kind and which file it wrote. The parent picks
        the destination (``handles/<id><ext>``), so no sandbox-supplied path is ever joined
        to the root. ``src`` is ``lstat``-ed and must be a regular file; it is renamed, not
        copied, so a multi-gigabyte save costs nothing. Call only when no sandboxed code is
        running. A payload that cannot be described is deleted and the call raises.
        """
        with self._lock:
            if not isinstance(id, str) or not HANDLE_ID_RE.match(id):
                raise ValueError(f"invalid handle id {id!r}: use letters, digits, '_', '-', "
                                 f"'.', starting with a letter or digit, at most 64 chars")
            ext = _KIND_EXT.get(kind) if isinstance(kind, str) else None
            if ext is None:
                raise ValueError(f"unknown handle kind: {kind!r}")
            rel = f"handles/{id}{ext}"
            dst = self.root / rel
            if id in self._handles or os.path.lexists(dst):
                raise HandleIdReuseError(id)
            try:
                st = os.lstat(src)
            except OSError as e:
                raise ValueError(f"handle payload is missing: {e}") from e
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("handle payload is not a regular file")
            os.rename(src, dst)
            try:
                os.chmod(dst, 0o644)
                described = self._describe(kind, dst)
            except Exception as e:  # noqa: BLE001 - a describer may raise anything on hostile bytes
                dst.unlink(missing_ok=True)
                raise ValueError(f"unreadable {kind} handle payload: {e}") from e
            handle = Handle(id=id, kind=kind, path=rel,
                            source=str(source)[:_MAX_SOURCE_CHARS], **described)
            self._handles[id] = handle
            self._advance_counter(id)
            self._save_manifest()
            return handle

    def put_input(self, *, path: str, source: str, input_id: str,
                  size: int | None = None, preview: str | None = None,
                  content_type: str | None = None, description: str | None = None,
                  replace: bool = False) -> Handle:
        """Register a host-ingested upload whose file already exists under ``inputs/``.

        Parent-authored (the host wrote the bytes), so it does not go through the sandbox
        adoption path. Idempotent by ``input_id``: an existing input is returned unchanged,
        unless ``replace`` is set, which overwrites its record and keeps its handle id.

        ``size`` and ``preview`` are accepted for call compatibility but **ignored**: like
        every other handle, an input's described fields are derived here from the bytes on
        disk, by the one describer that ``_register_record`` also uses on rehydration. That
        parity is what lets a reopened store reproduce the record exactly instead of
        believing the (child-writable) manifest.
        """
        with self._lock:
            existing = self.inputs().get(input_id)
            if existing is not None and not replace:
                return existing
            try:
                resolved = safe_path(self.root, path)
            except PathEscapesRootError as e:
                raise ValueError(f"input path escapes root: {path!r}") from e
            hid = existing.id if existing is not None else self._new_id()
            handle = Handle(id=hid, kind="binary", path=path, source=source, input_id=input_id,
                            description=description,
                            **self._describe_input(resolved, content_type))
            self._handles[handle.id] = handle
            self._save_manifest()
            return handle

    def _describe_input(self, path: Path, content_type: str | None) -> dict[str, Any]:
        """Derive an uploaded input's described fields from its bytes.

        Uploads are untrusted and are **never parsed**: the preview is the name/size/type
        line plus, for text-like extensions, the first few raw lines. ``content_type`` falls
        back to what the filename implies, so the value is reproducible from the file alone
        and a rehydrated record matches the one ``put_input`` wrote.
        """
        size = path.stat().st_size
        filename = path.name
        ctype = content_type or mimetypes.guess_type(filename)[0]
        head = f"<uploaded file {filename}, {size} bytes, {ctype or 'unknown type'}>"
        if Path(filename).suffix.lower() not in _TEXT_PREVIEW_EXTS:
            preview = head
        else:
            with path.open("rb") as f:
                raw = f.read(_PREVIEW_READ_BYTES)
            lines = raw.decode("utf-8", errors="replace").splitlines()[:_PREVIEW_LINES]
            preview = (head + "\n" + "\n".join(lines))[:_PREVIEW_CHARS]
        return {"bytes": size, "preview": preview, "content_type": ctype,
                "digest": _digest_file(path)}

    def inputs(self) -> dict[str, Handle]:
        """Uploaded inputs, keyed by host-supplied ``input_id``."""
        with self._lock:
            return {h.input_id: h for h in self._handles.values() if h.input_id is not None}

    @property
    def _manifest_file(self) -> Path:
        return self._state / "manifest.json"

    @property
    def _legacy_manifest_file(self) -> Path:
        return self.dir / "_manifest.json"

    def _save_manifest(self) -> None:
        """Persist {id: summary} atomically so a new HandleStore on this root can rehydrate.

        Lives under ``.sheath/``, which sandboxed code cannot write in the container tier.
        Rewrites the whole manifest on every mutation -- fine at handle frequency.
        """
        with self._lock:
            payload = json.dumps(self.manifest())
            self._atomic(self._manifest_file,
                         lambda tmp: tmp.write_text(payload, encoding="utf-8"))

    def _read_manifest_records(self, path: Path, max_bytes: int | None = None) -> dict | None:
        """Parse a manifest file. ``None`` if absent, a link, unreadable, malformed or (with
        ``max_bytes``) larger than that."""
        try:
            if max_bytes is not None:
                # No-follow, regular-file-only and size-capped: the legacy location was
                # writable by sandboxed code.
                raw = workspace.read_control_file(path.parent, path.name, max_bytes)
                if raw is None:
                    return None
                records = json.loads(raw.decode("utf-8"))
            else:
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    return None
                records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):   # incl. JSON/Unicode/ControlFileError
            return None
        return records if isinstance(records, dict) else None

    def _load_manifest(self) -> None:
        """Restore handles + the id counter from a prior session on this root.

        Nothing in a manifest is believed: each record is re-derived from its file.
        Tolerant by design -- a corrupt, forged, or orphaned record is skipped, never fatal.
        A manifest left at the pre-trust-zone location (``handles/_manifest.json``) is read
        once, rewritten at the new location and removed.
        """
        records = self._read_manifest_records(self._manifest_file)
        migrating = False
        if records is None:
            records = self._read_manifest_records(self._legacy_manifest_file,
                                                  max_bytes=_MAX_LEGACY_MANIFEST_BYTES)
            migrating = records is not None
        if records is None:
            return
        for record in records.values():
            try:
                self._register_record(record)
            except Exception:  # noqa: BLE001 - a describer may raise anything on a corrupt file
                continue
        if migrating:
            self._save_manifest()
            self._legacy_manifest_file.unlink(missing_ok=True)

    def verify(self, handle_id: str) -> Path:
        """Digest-check a handle and return its absolute path, without loading it."""
        try:
            handle = self._handles[handle_id]
        except KeyError:
            raise KeyError(f"no handle with id {handle_id!r}") from None
        path = safe_path(self.root, handle.path)  # defense-in-depth before any read
        self._verify_digest(handle, path)
        return path

    def get(self, handle_id: str) -> Any:
        path = self.verify(handle_id)
        handle = self._handles[handle_id]
        if handle.kind == "dataframe":
            import pandas as pd
            return pd.read_parquet(path)
        if handle.kind == "json":
            return json.loads(path.read_text(encoding="utf-8"))
        if handle.kind == "binary":
            return str(path)  # binary content is opened by a library; hand back the path
        return path.read_text(encoding="utf-8")

    def _verify_digest(self, handle: Handle, path: Path) -> None:
        """Fail closed if a handle's bytes changed since its metadata was derived.

        Metadata is derived once, at creation. Nothing stopped a later ``run_python`` from
        overwriting the bytes under an existing handle without touching the control channel,
        leaving the model holding a summary of data no longer on disk. Checked on the parent
        side of every read, including ``binary`` handles -- those hand a path to a library
        that is about to read it, which is the same exposure.

        ``digest is None`` means nothing was recorded, so there is nothing to compare
        against and the read proceeds. See ``_digest_file`` for what the digest does and
        does not cover.
        """
        if handle.digest is None:
            return
        if _digest_file(path) != handle.digest:
            raise HandleTamperedError(handle.id, handle.path)

    def summary(self, handle_id: str) -> dict[str, Any]:
        with self._lock:
            return self._handles[handle_id].summary()

    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return {hid: h.summary() for hid, h in self._handles.items()}

    def manifest_handles(self) -> dict[str, Handle]:
        with self._lock:
            return dict(self._handles)
