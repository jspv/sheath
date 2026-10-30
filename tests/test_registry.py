import inspect

from sheath import SheathConfig, Session
from sheath.tools.registry import build_tools


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


def test_build_tools_returns_expected_named_callables(tmp_path):
    tools = build_tools(_session(tmp_path))
    names = {t.__name__ for t in tools}
    assert names == {
        "read_file", "write_file", "edit_file", "list_files", "search",
        "fetch_url", "run_python", "inspect_handle",
        "web_search", "web_extract", "read_document",
        "publish_file", "publish_handle",   # filtered by Session.tools() without on_publish
    }


def test_build_tools_includes_web_tools(tmp_path):
    names = {t.__name__ for t in build_tools(_session(tmp_path))}
    assert {"web_search", "web_extract"} <= names


def test_build_tools_includes_read_document(tmp_path):
    tools = {t.__name__: t for t in build_tools(_session(tmp_path))}
    assert "read_document" in tools
    params = list(inspect.signature(tools["read_document"]).parameters)
    assert params == ["source"]


def test_wrapped_tools_do_not_expose_session_param(tmp_path):
    tools = {t.__name__: t for t in build_tools(_session(tmp_path))}
    assert "session" not in inspect.signature(tools["read_file"]).parameters
    assert list(inspect.signature(tools["read_file"]).parameters)[0] == "path"


def test_wrapped_tools_keep_docstrings(tmp_path):
    tools = {t.__name__: t for t in build_tools(_session(tmp_path))}
    assert tools["search"].__doc__ and "pattern" in tools["search"].__doc__.lower()


def test_wrapped_tools_actually_work(tmp_path):
    sess = _session(tmp_path)
    tools = {t.__name__: t for t in build_tools(sess)}
    tools["write_file"]("a.txt", "hello\nworld\n")
    assert tools["read_file"]("a.txt") == "hello\nworld\n"
    out = tools["run_python"](code="from sheath_sandbox import emit\nemit(5)\n")
    assert out["result"] == 5
    # search via the wrapped closure (regression: the package re-export shadowed the
    # `search` submodule, so the closure called the function as if it were a module).
    hits = tools["search"]("world", path="a.txt")
    assert hits and hits[0]["line"] == 2
    assert tools["list_files"](".") and "a.txt" in tools["list_files"](".")


def test_wrapped_file_tools_forward_every_parameter(tmp_path):
    # The model calls the wrappers, not the impls: each new parameter must reach the impl.
    tools = {t.__name__: t for t in build_tools(_session(tmp_path))}
    tools["write_file"]("d/a.txt", "one\ntwo\nthree\n")
    tools["write_file"]("b.csv", "x\n")
    assert tools["read_file"]("d/a.txt", start_line=2, limit=1, line_numbers=True) == (
        "     2\ttwo\n")
    assert tools["list_files"](".", pattern="**/*.txt") == ["d/a.txt"]
    [hit] = tools["search"]("two", path="d", context=1)
    assert hit["before"] == ["one"] and hit["after"] == ["three"]
    assert tools["search"]("t", output="count") == [{"file": "d/a.txt", "count": 2}]
    assert tools["search"]("e", output="files", max_matches=5) == [{"file": "d/a.txt"}]
    out = tools["edit_file"]("d/a.txt", "o", "0", replace_all=True)
    assert out.startswith("replaced 2 occurrence(s)")
    assert tools["read_file"]("d/a.txt") == "0ne\ntw0\nthree\n"
