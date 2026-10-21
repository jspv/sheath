"""Executes a user script with Jupyter-style last-expression auto-capture.

The sandbox runs ``python _runner.py <user_script> <args...>`` instead of the user
script directly. If the script's last top-level statement is a bare expression (e.g.
``fib_12`` or ``df.head()``), its value is auto-emitted as the result -- so the model
doesn't have to know about ``emit()`` to get a return value. An explicit ``emit()``
(or any prior write to the emit file) always wins; ``print(...)`` output is still
captured as stdout (and the sandbox falls back to stdout when nothing was emitted).
"""

import ast
import os
import sys

import sheath_sandbox as _hs


def _emit_already_written() -> bool:
    path = os.path.join(os.environ.get("SHEATH_RUN_DIR", ""), "emit.json")
    return os.path.exists(path) and os.path.getsize(path) > 0


def main() -> None:
    script = sys.argv[1]
    sys.argv = [script, *sys.argv[2:]]      # present argv as if the user script ran directly
    sys.path.insert(0, os.getcwd())          # let the user script import sibling modules

    with open(script, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=script)
    last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None

    # The model is told load/save/emit (and publish, when the host accepts files) need no import.
    namespace = {"__name__": "__main__", "__file__": script,
                 "load": _hs.load, "save": _hs.save, "emit": _hs.emit}
    if os.environ.get("SHEATH_PUBLISH_ENABLED") == "1":
        namespace["publish"] = _hs.publish
    exec(compile(tree, script, "exec"), namespace)

    if last_expr is not None:
        value = eval(compile(ast.Expression(last_expr.value), script, "eval"), namespace)
        if value is not None and not _emit_already_written():
            _hs.emit(value)


main()
