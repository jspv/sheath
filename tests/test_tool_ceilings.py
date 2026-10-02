"""No built-in tool result may exceed a hard ceiling, whatever the workspace holds."""

import json

import pandas as pd
import pytest

from tether import Session, TetherConfig
from tether.tools.registry import build_tools

_CEILING = 64_000   # bytes of JSON; well under any model context window


@pytest.fixture
def hostile(tmp_path):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"))
    (sess.root / "one_line.txt").write_text("x" * 5_000_000)
    for i in range(3000):
        (sess.root / f"f{i:04d}.txt").write_text("needle " + "y" * 2000 + "\n")
    text = sess.store.put("B" * 3_000_000, source="s")
    frame = sess.store.put(pd.DataFrame({"a": ["v" * 50_000] * 200}), source="s")
    return sess, text.id, frame.id


def _tool(sess, name):
    return next(t for t in build_tools(sess) if t.__name__ == name)


@pytest.mark.parametrize("call", [
    lambda s, t, f: _tool(s, "read_file")("one_line.txt", 0, 10**9),
    lambda s, t, f: _tool(s, "list_files")("."),
    lambda s, t, f: _tool(s, "search")("needle", ".", None, False, 10**9),
    lambda s, t, f: _tool(s, "run_python")("print('A' * 5_000_000)"),
    lambda s, t, f: _tool(s, "run_python")(
        "from tether_sandbox import emit\nemit('Z' * 500_000)"),
    lambda s, t, f: _tool(s, "inspect_handle")(t, 10**9, True),
    lambda s, t, f: _tool(s, "inspect_handle")(f, 10**9, True),
])
def test_builtin_tool_results_stay_under_the_ceiling(hostile, call):
    sess, text_id, frame_id = hostile
    result = call(sess, text_id, frame_id)
    assert len(json.dumps(result, default=str).encode()) <= _CEILING
