"""publish_file / publish_handle: deliver finished workspace files to the host.

Both go through ``Session.publish``, which validates the file and invokes the host callback;
the model only ever sees the returned metadata, never the file bytes.
"""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path

from .. import workspace
from ..handles import HandleTamperedError
from ..paths import safe_filename
from ..sandbox import SandboxQuarantinedError
from ..session import Session, _ensure_real_dir

_OUTPUTS = "outputs"
_DATAFRAME_FORMATS = ("csv", "xlsx", "parquet", "json")

# Runs inside the sandbox. Fixed text: the handle id, format and output path arrive as
# argv, never by string interpolation, so nothing model-supplied is ever executed as code.
_CONVERT_SCRIPT = """\
import sys
from sheath_sandbox import load

handle_id, fmt, out = sys.argv[1:4]
df = load(handle_id)
if fmt == "csv":
    df.to_csv(out, index=False)
elif fmt == "xlsx":
    df.to_excel(out, index=False)
elif fmt == "parquet":
    df.to_parquet(out)
else:
    df.to_json(out, orient="records")
"""


def publish_file(session: Session, path: str, name: str | None = None,
                 description: str | None = None) -> dict:
    """Deliver a finished workspace file to the user; returns its metadata + the host reply."""
    return session.publish(path, name=name, description=description, source="tool:publish_file")


def publish_handle(session: Session, handle_id: str, format: str | None = None,
                   name: str | None = None, description: str | None = None) -> dict:
    """Write a handle to ``outputs/`` (converting a dataframe to ``format``) and publish it.

    Holds the session's sandbox lock throughout, so no sandboxed code can swap ``outputs/``
    between the containment check and the write.
    """
    try:
        with session.file_io():
            return _publish_handle(session, handle_id, format, name, description)
    except SandboxQuarantinedError as e:
        return {"error": str(e)}


def _publish_handle(session: Session, handle_id: str, format: str | None, name: str | None,
                    description: str | None) -> dict:
    handle = session.store.manifest_handles().get(handle_id)
    if handle is None:
        return {"error": f"no handle with id {handle_id!r}"}
    try:
        if handle.kind == "dataframe":
            fmt = (format or "csv").lower()
            if fmt not in _DATAFRAME_FORMATS:
                return {"error": f"unsupported format {format!r}; use one of {_DATAFRAME_FORMATS}"}
            filename = safe_filename(name, fallback=f"{handle_id}.{fmt}")
            if not filename.lower().endswith(f".{fmt}"):
                filename += f".{fmt}"
            session.store.verify(handle_id)        # digest check; the bytes are not loaded here
            rel = _convert_in_sandbox(session, handle_id, fmt, filename)
        else:
            backing = session.store.verify(handle_id)
            ext = backing.suffix.lstrip(".").lower()
            if format is not None and format.lower() != ext:
                return {"error": f"a {handle.kind} handle is delivered as-is (.{ext}); "
                                 "format conversion applies to dataframe handles only"}
            filename = safe_filename(name, fallback=backing.name)
            rel = _write_output(session, filename, lambda p: shutil.copyfile(backing, p))
    except (OSError, ValueError, ImportError, HandleTamperedError) as e:
        return {"error": f"could not write {handle_id} for publishing: {e}"}
    return session.publish(rel, name=filename, description=description,
                           source="tool:publish_handle")


def _outputs_dir(session: Session) -> Path:
    """Return ``<root>/outputs`` as a REAL directory. The caller holds the session lock.

    Sandboxed code can replace ``outputs/`` with anything, including a link to a protected
    directory (``outputs -> handles``) that ``safe_path`` would happily resolve inside the
    root. So the name is never followed: a link or non-directory found there is removed and
    replaced by a real directory, exactly as ``inputs/<id>`` is repaired.
    """
    outputs = session.root / _OUTPUTS
    _ensure_real_dir(outputs)
    if workspace.is_protected(session.root, outputs.resolve()):
        raise ValueError("outputs/ is a sheath-managed directory; refusing to write there")
    return outputs


def _convert_in_sandbox(session: Session, handle_id: str, fmt: str, filename: str) -> str:
    """Convert a dataframe handle to ``outputs/<filename>`` by running a fixed script in the
    sandbox, so the parent never parses or materializes sandbox-authored parquet.

    The script writes to a name the parent just invented (so nothing can be pre-planted
    there); once it has exited, the parent checks the result is a regular file and renames
    it over the destination, which replaces a planted link instead of writing through it.
    The caller holds the session lock.
    """
    outputs = _outputs_dir(session)
    tmp_name = f".publish_{secrets.token_hex(8)}.{fmt}"
    tmp = outputs / tmp_name
    try:
        res = session.sandbox.run_code(_CONVERT_SCRIPT,
                                       [handle_id, fmt, f"{_OUTPUTS}/{tmp_name}"])
        if res.exit_code != 0:
            lines = [ln for ln in (res.error or "").splitlines() if ln.strip()]
            raise ValueError(lines[-1] if lines else f"conversion exited {res.exit_code}")
        if not stat.S_ISREG(os.lstat(tmp).st_mode):
            raise ValueError("conversion did not produce a regular file")
        os.replace(tmp, outputs / filename)
    finally:
        if os.path.lexists(tmp) and not os.path.isdir(tmp):
            tmp.unlink()
    return f"{_OUTPUTS}/{filename}"


def _write_output(session: Session, filename: str, write) -> str:
    """Write ``outputs/<filename>`` without ever following a link sandboxed code planted.

    ``outputs/`` itself is made a real directory first (a planted link there is replaced,
    never followed), the bytes go to a fresh temp file in it, and ``os.replace`` swaps that in, which
    replaces a planted symlink at the destination instead of writing through it.
    """
    outputs = _outputs_dir(session)
    fd, tmp = tempfile.mkstemp(dir=outputs, prefix=".publish_")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, outputs / filename)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return f"{_OUTPUTS}/{filename}"
