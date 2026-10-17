"""Tool status updates: a sheath-owned side-band channel.

A tool calls ``report_progress(...)`` while it runs; the call routes through a
``contextvars`` lookup to the ``StatusBus`` the active ``Session`` bound for this run, which
fans the event out to subscribers (e.g. a ``Sheath(on_status=...)`` callback or the CLI
``--verbose`` printer). Outside a bound run, ``report_progress`` is a no-op, so tools can
call it unconditionally. This is a side-band channel: mid-tool and MCP progress are not part
of MAF's response stream (which carries text deltas and tool-call lifecycle), so the sheath
delivers them itself.
"""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

_log = logging.getLogger("sheath.status")


@dataclass(frozen=True)
class StatusEvent:
    tool: str                       # emitting tool name, or "sheath"
    message: str                    # human-readable status line
    current: float | None = None    # progress numerator (optional)
    total: float | None = None      # progress denominator (optional)
    seq: int = 0                    # monotonic per-bus ordering (filled by the bus)
    timestamp: float = 0.0          # wall-clock, time.time() (filled by the bus)


class StatusBus:
    """Thread-safe fan-out of StatusEvents to subscriber callbacks."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[StatusEvent], None]] = []
        self._lock = threading.Lock()
        self._seq = 0

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> Callable[[], None]:
        """Register ``callback``; returns a zero-arg handle that unsubscribes it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return unsubscribe

    def emit(self, event: StatusEvent) -> None:
        """Stamp ``event`` with seq + timestamp and deliver to every subscriber.

        Subscribers are snapshotted under the lock, then called outside it (so a subscriber
        may itself emit without deadlocking). A raising subscriber is swallowed -- status is
        best-effort and must never break the task.
        """
        with self._lock:
            self._seq += 1
            stamped = dataclasses.replace(event, seq=self._seq, timestamp=time.time())
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(stamped)
            except Exception:  # noqa: BLE001 - best-effort; never propagate into the task
                _log.debug("status subscriber raised", exc_info=True)


_current: contextvars.ContextVar[StatusBus | None] = contextvars.ContextVar(
    "sheath_status_bus", default=None
)


def current_bus() -> StatusBus | None:
    return _current.get()


@contextmanager
def bind_bus(bus: StatusBus) -> Iterator[None]:
    """Make ``bus`` the target of ``report_progress`` for the duration of the block.

    Restores the previous value with ``set`` rather than ``reset(token)`` so the block may be
    entered and exited in *different* asyncio tasks/contexts: a continuous ``Session`` binds on
    open (in one request's task) and unbinds on close (in another), where ``reset(token)`` would
    raise "Token was created in a different Context". Same-task use restores identically.
    """
    prev = _current.get()
    _current.set(bus)
    try:
        yield
    finally:
        _current.set(prev)


def report_progress(message: str, *, current: float | None = None,
                    total: float | None = None, tool: str = "tool") -> None:
    """Emit a status update from inside a tool. No-op outside a bound run."""
    bus = _current.get()
    if bus is None:
        return
    bus.emit(StatusEvent(tool=tool, message=message, current=current, total=total))
