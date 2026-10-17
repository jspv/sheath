"""Child-process half of the Python search fallback.

The pattern is model-supplied and the files are model-written, so a catastrophic regex can
backtrack for longer than the universe has existed -- inside one C call that holds the GIL.
Run in-process that would freeze the whole host; here it runs in a child that
``procio.run_bounded`` kills on timeout.

Invoked as ``python -I -m sheath.tools._search_worker <request.json>``; the request names
the root, base, pattern, flags, glob and match cap. Prints one JSON object on stdout:
``{"hits": [...]}`` or ``{"error": "..."}``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path

from sheath import workspace
from sheath.tools.search import (_MAX_LINE_CHARS as MAX_LINE_CHARS,
                                 _MAX_SEARCH_FILE_BYTES as MAX_SEARCH_FILE_BYTES,
                                 _snippet as snippet)


def scan(root: Path, base: Path, pattern: str, ignore_case: bool, max_matches: int,
         glob: str | None = None) -> list[dict]:
    """The confined scan: never reads outside the root, skips links, special files and
    files over 50 MiB, and bounds every line. Raises ValueError for an invalid pattern."""
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:  # match the rg path: invalid regex -> ValueError
        raise ValueError(f"invalid search pattern: {e}") from e
    files = [base] if base.is_file() else sorted(base.rglob(glob or "*"))
    hits: list[dict] = []
    real_root = root.resolve()
    for f in files:
        try:
            if not f.resolve().is_relative_to(real_root):
                continue                   # never read outside the root (links, odd globs)
            st = os.lstat(f)               # lstat: a link is skipped, never followed
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_SEARCH_FILE_BYTES:
                continue
            fd = os.open(f, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                continue
            with open(fd, encoding="utf-8", newline="") as fh:
                for lineno, line in enumerate(
                        workspace.iter_bounded_lines(fh, MAX_LINE_CHARS), start=1):
                    m = rx.search(line)
                    if m:
                        hits.append({"file": f.relative_to(root).as_posix(), "line": lineno,
                                     "col": m.start(), "text": snippet(line, m.start())})
                        if len(hits) >= max_matches:
                            return hits
        except (UnicodeDecodeError, OSError):
            continue
    return hits


def main(argv: list[str]) -> int:
    req = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        hits = scan(Path(req["root"]), Path(req["base"]), req["pattern"],
                    bool(req["ignore_case"]), int(req["max_matches"]), req.get("glob"))
    except ValueError as e:
        out = {"error": str(e)}
    else:
        out = {"hits": hits}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
