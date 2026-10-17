"""run_python: execute agent-authored Python in the sandbox and return a result dict."""

from __future__ import annotations

import json
from dataclasses import asdict

from ..session import Session
from ..status import report_progress


def run_python(session: Session, code: str | None = None, path: str | None = None,
               args: list[str] | None = None) -> dict:
    """Run Python in the sandbox. Provide ``path`` (a script file under the root) or
    ``code`` (inline, written to a scratch script then run). Scripts may use the injected
    ``load(id)`` / ``save(id, obj)`` / ``emit(obj)`` / ``publish(path)`` helpers. Returns the
    ExecResult fields: ``stdout, stderr, result, error, exit_code, new_handles, killed_by,
    published``. ``stdout``/``stderr`` are capped; an emitted ``result`` larger than the
    spill threshold is stored as a handle and replaced by its summary.
    """
    report_progress("running script in sandbox", tool="run_python")
    if path is not None:
        res = session.sandbox.run_script(path, args)
    elif code is not None:
        res = session.sandbox.run_code(code, args)
    else:
        raise ValueError("run_python requires either code= or path=")
    out = asdict(res)
    result = out["result"]
    # The stdout fallback is already capped by the sandbox; only a deliberate emit() (or a
    # final expression) can be large, and that is data, so it becomes a handle.
    from_stdout = isinstance(result, str) and result == out["stdout"].strip()
    if result is not None and not from_stdout:
        encoded = json.dumps(result, default=str)
        if len(encoded.encode()) > session.config.spill_threshold_bytes:
            payload = result if isinstance(result, (dict, list, str)) else encoded
            handle = session.store.put(payload, source="run_python:result")
            out["result"] = {
                "spilled": True,
                "note": "the result was large, so it was stored as a handle; use "
                        "inspect_handle or load() it in run_python",
                "handle": handle.summary(),
            }
            out["new_handles"].append(handle.id)
    return out
