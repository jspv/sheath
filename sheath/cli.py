"""Thin CLI over Sheath.solve()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .api import Sheath
from .config import SheathConfig
from .sandbox import SandboxRuntimeUnavailable
from .status import StatusEvent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sheath", description="Run the data-integration sheath.")
    p.add_argument("problem", help="The task to solve.")
    p.add_argument("--model", default="gpt-5-mini",
                   help="Model for the built-in OpenAI client (the CLI uses that client; "
                        "to use another provider, drive Sheath from Python with your own client).")
    p.add_argument("--root", default=None, help="Workspace root dir (default: a fresh session dir).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print live tool status to stderr as the task runs.")
    return p


def make_status_printer(write: Callable[[str], None] | None = None):
    """A status sink for --verbose: formats each StatusEvent to a line."""
    if write is None:
        def write(line: str) -> None:
            print(line, file=sys.stderr)

    def on_status(event: StatusEvent) -> None:
        progress = ""
        if event.current is not None and event.total is not None:
            progress = f" [{event.current:g}/{event.total:g}]"
        write(f"→ {event.tool}: {event.message}{progress}")

    return on_status


def run_cli(argv: list[str] | None = None, client=None) -> int:
    args = build_parser().parse_args(argv)
    on_status = make_status_printer() if args.verbose else None
    cfg = SheathConfig(model=args.model,
                        root_dir=Path(args.root) if args.root else None)
    result = Sheath(cfg, client=client).solve(args.problem, on_status=on_status)
    if result.final_text:
        print(result.final_text)
    if result.error:
        print(f"\n[run did not complete: {result.error}]")
    print(f"\n[session: {result.session_dir}]")
    return 1 if result.error else 0


def main() -> None:
    """Console-script entry. Exit 2 on an unusable sandbox, with the message, not a traceback.

    ``SandboxRuntimeUnavailable`` is a configuration problem with a remedy spelled out in
    its message (install a runtime, or opt into the local tier). A stack trace buries that
    and reads like a crash in the harness.
    """
    try:
        code = run_cli()
    except SandboxRuntimeUnavailable as e:
        print(str(e), file=sys.stderr)
        code = 2
    raise SystemExit(code)
