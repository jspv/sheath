"""Run a child process with bounded output capture.

``subprocess.run(capture_output=True)`` buffers everything the child prints. The child here
is model-authored code, so each stream is drained through a buffer that keeps only the
first and last halves of a byte cap and counts what it dropped.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 65536
_JOIN_TIMEOUT_S = 5.0


class _HeadTail:
    """Keeps the first ``cap // 2`` and last ``cap - cap // 2`` bytes fed to it."""

    def __init__(self, cap: int) -> None:
        self._head_cap = cap // 2
        self._tail_cap = cap - self._head_cap
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0

    def feed(self, chunk: bytes) -> None:
        self._total += len(chunk)
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                del self._tail[:excess]

    def text(self) -> str:
        dropped = self._total - len(self._head) - len(self._tail)
        if dropped <= 0:
            return bytes(self._head + self._tail).decode("utf-8", errors="replace")
        return (self._head.decode("utf-8", errors="replace")
                + f"\n... [tether: {dropped} bytes of output omitted] ...\n"
                + self._tail.decode("utf-8", errors="replace"))


@dataclass
class BoundedRun:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def _drain(stream, buf: _HeadTail) -> None:
    try:
        while chunk := stream.read1(_CHUNK):
            buf.feed(chunk)
    finally:
        stream.close()


def run_bounded(argv: list[str], *, timeout: float, cap: int,
                cwd: Path | str | None = None, env: dict[str, str] | None = None,
                preexec_fn: Callable[[], None] | None = None) -> BoundedRun:
    """Run ``argv``; return its exit code and at most ``cap`` bytes of each output stream.

    On timeout the process is killed and whatever was captured is returned with
    ``timed_out=True``. The drain threads are daemons and are joined with a bound, so a
    grandchild that keeps a pipe open cannot hang the caller.
    """
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=preexec_fn)
    out, err = _HeadTail(cap), _HeadTail(cap)
    threads = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    return BoundedRun(out.text(), err.text(), proc.returncode, timed_out)
