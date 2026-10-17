"""Text search over the session root: ripgrep when available, else a Python regex scan in
a bounded child process."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from ..paths import safe_path
from ..procio import run_bounded
from ..session import Session

_MAX_MATCHES = 1000
_MAX_SEARCH_FILE_BYTES = 50 * 1024 * 1024
_MAX_LINE_CHARS = 1_000_000
_RG_TIMEOUT_S = 30.0
_PY_TIMEOUT_S = _RG_TIMEOUT_S
_PY_OUTPUT_CAP = 16 * 1024 * 1024   # 1000 hits of bounded snippets fit with room to spare

_MAX_SEARCH_RESULT_CHARS = 60_000

_SNIPPET_WIDTH = 240  # cap each hit's text so a single huge line can't flood context


def _snippet(line: str, col: int) -> str:
    """A bounded window of ``line`` centered on ``col`` (minified one-line files can be
    hundreds of KB; returning the whole matching line would blow the model's context)."""
    line = line.rstrip("\n")
    if len(line) <= _SNIPPET_WIDTH:
        return line
    start = max(0, col - _SNIPPET_WIDTH // 2)
    end = start + _SNIPPET_WIDTH
    return ("…" if start > 0 else "") + line[start:end] + ("…" if end < len(line) else "")


def _fit(hits: list[dict]) -> list[dict]:
    """Keep the longest prefix of ``hits`` whose JSON stays under the result budget."""
    size = 0
    for i, hit in enumerate(hits):
        size += len(json.dumps(hit)) + 2
        if size > _MAX_SEARCH_RESULT_CHARS:
            return hits[:i] + [{"note": f"{len(hits) - i} more matches omitted; narrow the "
                                        "pattern or the path"}]
    return hits


def _check_glob(glob: str | None) -> None:
    if glob and (glob.startswith(("/", "\\")) or ".." in re.split(r"[/\\]", glob)
                 or re.match(r"^[A-Za-z]:", glob)):
        raise ValueError(f"invalid glob {glob!r}: must be relative and contain no '..'")


def _rg_bytes(obj: dict) -> bytes:
    """rg --json gives {"text": ...} for valid UTF-8, else {"bytes": <base64>}."""
    if "text" in obj:
        return obj["text"].encode("utf-8")
    return base64.b64decode(obj.get("bytes", ""))


def search(session: Session, pattern: str, path: str = ".", glob: str | None = None,
           ignore_case: bool = False, max_matches: int = 100) -> list[dict]:
    """Search for ``pattern`` (a regex) in files under ``path`` (a file or folder).

    Returns up to ``max_matches`` (at most 1000) hits as ``{"file", "line", "col", "text"}``
    with ``file`` relative to the session root. Uses ripgrep if installed, else a Python
    scan. Links, special files and files over 50 MiB are skipped.
    """
    max_matches = max(1, min(int(max_matches), _MAX_MATCHES))
    _check_glob(glob)
    with session.file_io():
        base = safe_path(session.root, path or ".")  # "" / None -> the root
        mode = os.stat(base).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"not a regular file or directory: {path!r}")
        rg = shutil.which("rg")
        if rg:
            return _fit(_search_rg(rg, session.root, base, pattern, glob, ignore_case,
                                   max_matches))
        return _fit(_search_python(session.root, base, pattern, ignore_case, max_matches,
                                   glob))


def _search_rg(rg: str, root: Path, base: Path, pattern: str, glob: str | None,
               ignore_case: bool, max_matches: int) -> list[dict]:
    # ripgrep does not follow symlinks and skips non-regular files when walking a directory.
    cmd = [rg, "--json", "--no-heading", "--max-filesize", str(_MAX_SEARCH_FILE_BYTES)]
    if ignore_case:
        cmd.append("-i")
    if glob:
        cmd += ["-g", glob]
    # Run in the root with a root-relative base, so neither rg's error text nor its result
    # paths ever contain the absolute host path of the session root.
    rel_base = base.relative_to(root).as_posix() if base != root else "."
    cmd += ["--", pattern, rel_base]
    hits: list[dict] = []
    stopped_early = False
    # stderr goes to a file so a noisy rg can never fill a pipe and deadlock the read loop.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=errf, text=True)
        timer = threading.Timer(_RG_TIMEOUT_S, proc.kill)
        timer.start()
        try:
            for line in proc.stdout:           # streamed: stop reading once we have enough
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") != "match":
                    continue
                data = evt["data"]
                abs_path = Path(_rg_bytes(data["path"]).decode("utf-8", errors="replace"))
                rel = (abs_path.relative_to(root).as_posix() if abs_path.is_absolute()
                       else abs_path.as_posix())
                # One hit per matching line (first submatch), with a CHARACTER column, to
                # match the Python fallback exactly. rg reports a byte offset, so convert it.
                raw = _rg_bytes(data["lines"])
                line_text = raw.decode("utf-8", errors="replace")
                sub = data["submatches"][0]
                char_col = len(raw[:sub["start"]].decode("utf-8", errors="replace"))
                hits.append({"file": rel, "line": data["line_number"], "col": char_col,
                             "text": _snippet(line_text, char_col)})
                if len(hits) >= max_matches:
                    stopped_early = True
                    proc.kill()
                    break
        finally:
            timer.cancel()
            proc.stdout.close()
            proc.wait()
        # rg exit codes: 0 = match, 1 = no match, 2 = error (e.g. invalid regex).
        if not stopped_early and proc.returncode == 2:
            errf.seek(0)
            message = errf.read(4096).decode("utf-8", errors="replace").strip()
            if not hits:
                raise ValueError(message or "ripgrep search failed")
        if not stopped_early and proc.returncode < 0:
            raise ValueError(f"search timed out after {_RG_TIMEOUT_S:.0f}s; narrow the path "
                             "or the pattern")
    return hits


def _search_python(root: Path, base: Path, pattern: str, ignore_case: bool,
                   max_matches: int, glob: str | None = None) -> list[dict]:
    """The fallback scan, run in a bounded child process (``_search_worker``).

    A model-supplied catastrophic regex against a model-written file backtracks inside one
    C call holding the GIL; in-process that freezes the whole host. The child is killed
    after ``_PY_TIMEOUT_S`` and its output is capped. The request travels in a private temp
    file (argv has a length limit; stdin is closed).
    """
    request = {"root": str(root), "base": str(base), "pattern": pattern,
               "ignore_case": ignore_case, "max_matches": max_matches, "glob": glob}
    # Outside the root (the host's private temp dir, mode 0600): inside it, the scan would
    # find the request file -- which contains the pattern -- and report it as a hit.
    fd, req_path = tempfile.mkstemp(prefix="sheath-search-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(request, f)
        run = run_bounded([sys.executable, "-I", "-m", "sheath.tools._search_worker",
                           req_path], timeout=_PY_TIMEOUT_S, cap=_PY_OUTPUT_CAP,
                          cwd=tempfile.gettempdir())
    finally:
        os.unlink(req_path)
    if run.timed_out:
        raise ValueError(f"search timed out after {_PY_TIMEOUT_S:.0f}s; narrow the path "
                         "or the pattern")
    try:
        out = json.loads(run.stdout)
    except json.JSONDecodeError:
        out = None
    if run.returncode != 0 or not isinstance(out, dict):
        tail = run.stderr.strip().splitlines()[-1:] or [f"exit {run.returncode}"]
        raise ValueError(f"search failed: {tail[0]}")
    if "error" in out:
        raise ValueError(str(out["error"]))
    return list(out.get("hits", []))[:max_matches]
