"""TETHER-4: publish_file / publish_handle / sandbox publish() deliver files to the host."""

import asyncio
import contextvars
import hashlib
import json
import os

import pandas as pd
import pytest

from tether import Session, Tether, TetherConfig
from tether.publish import PublishedFile
from tether.testing import StubChatClient, text, tool_call


class _Host:
    def __init__(self, result=None, raises=None):
        self.calls: list[PublishedFile] = []
        self.copies: dict[str, bytes] = {}
        self.result = result if result is not None else {"file_id": "F1"}
        self.raises = raises

    def __call__(self, pf: PublishedFile):
        self.calls.append(pf)
        self.copies[pf.name] = pf.path.read_bytes()   # the host copies during the callback
        if self.raises:
            raise self.raises
        return self.result


def _session(tmp_path, host=None, **cfg):
    return Session.create(TetherConfig(root_dir=tmp_path / "r", **cfg), on_publish=host)


def _write(sess, rel, data=b"x,y\n1,2\n"):
    p = sess.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _tool(sess, name):
    return next(t for t in sess.tools("code", "deliver") if t.__name__ == name)


# --- publish_file -------------------------------------------------------------------------

def test_publish_file_calls_host_once_with_size_and_sha(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    data = b"x,y\n1,2\n"
    _write(sess, "outputs/a.csv", data)
    out = _tool(sess, "publish_file")("outputs/a.csv", description="the table")
    assert len(host.calls) == 1
    pf = host.calls[0]
    assert pf.size == len(data) and pf.sha256 == hashlib.sha256(data).hexdigest()
    assert pf.source == "tool:publish_file" and pf.description == "the table"
    assert host.copies["a.csv"] == data
    assert out["host"] == {"file_id": "F1"}
    assert out["name"] == "a.csv" and out["rel_path"] == "outputs/a.csv"
    assert out["size"] == len(data) and out["sha256"] == pf.sha256
    assert (sess.root / "outputs/a.csv").read_bytes() == data   # untouched by tether


@pytest.mark.parametrize("setup, path", [
    (lambda s, t: None, "../x"),
    (lambda s, t: None, "/etc/hosts"),
    (lambda s, t: os.symlink("/etc/hosts", s.root / "outputs" / "l.csv"), "outputs/l.csv"),
    (lambda s, t: (s.root / "outputs" / "d").mkdir(), "outputs/d"),
    (lambda s, t: (s.root / "handles" / "h9.json").write_text("{}"), "handles/h9.json"),
    (lambda s, t: (s.root / "outputs" / "big").write_bytes(b"x" * 2048), "outputs/big"),
])
def test_publish_file_rejections_never_call_host(tmp_path, setup, path):
    host = _Host()
    sess = _session(tmp_path, host, max_publish_bytes=1024)
    (sess.root / "outputs").mkdir()
    (tmp_path / "x").write_text("outside")
    setup(sess, tmp_path)
    out = _tool(sess, "publish_file")(path)
    assert "error" in out
    assert host.calls == []


def test_host_callback_raising_returns_error(tmp_path):
    sess = _session(tmp_path, _Host(raises=RuntimeError("storage full")))
    _write(sess, "outputs/a.csv")
    out = _tool(sess, "publish_file")("outputs/a.csv")
    assert "storage full" in out["error"]
    assert "host" not in out


def test_publish_emits_status_event(tmp_path):
    sess = _session(tmp_path, _Host())
    events = []
    sess.subscribe(events.append)
    _write(sess, "outputs/a.csv")
    _tool(sess, "publish_file")("outputs/a.csv")
    assert any(e.tool == "publish_file" and "a.csv" in e.message for e in events)


def test_deliver_tools_hidden_without_callback(tmp_path):
    sess = _session(tmp_path, None)
    names = {t.__name__ for t in sess.tools("code", "deliver")}
    assert "publish_file" not in names and "publish_handle" not in names
    assert "publish_file" not in sess.tether_instructions("code", "deliver")


def test_deliver_tools_and_instructions_with_callback(tmp_path):
    sess = _session(tmp_path, _Host())
    names = {t.__name__ for t in sess.tools("code", "deliver")}
    assert {"publish_file", "publish_handle"} <= names
    instr = sess.tether_instructions("code", "deliver")
    assert "outputs/" in instr and "publish_file" in instr and "publish(" in instr


def test_deliver_only_bundle_without_callback_does_not_expand_to_all(tmp_path):
    sess = _session(tmp_path, None)
    assert {t.__name__ for t in sess.tools("deliver")} == {"inspect_handle"}


# --- publish_handle -----------------------------------------------------------------------

def test_publish_handle_dataframe_as_csv(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1, 2]}), source="t")
    out = _tool(sess, "publish_handle")(h.id, format="csv", name="result")
    assert "error" not in out, out
    assert out["name"] == "result.csv" and out["rel_path"] == "outputs/result.csv"
    assert host.copies["result.csv"] == b"a\n1\n2\n"
    assert host.calls[0].source == "tool:publish_handle"


def test_publish_handle_parquet_and_json(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    assert "error" not in _tool(sess, "publish_handle")(h.id, format="parquet")
    out = _tool(sess, "publish_handle")(h.id, format="json")
    assert json.loads(host.copies[out["name"]]) == [{"a": 1}]


def test_publish_handle_rejects_unknown_format(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    assert "error" in _tool(sess, "publish_handle")(h.id, format="exe")
    assert "error" in _tool(sess, "publish_handle")("h999")
    assert host.calls == []


def test_publish_handle_copies_non_dataframe_as_is(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put("hello", source="t")
    out = _tool(sess, "publish_handle")(h.id, name="note.txt")
    assert host.copies["note.txt"] == b"hello"
    assert out["rel_path"] == "outputs/note.txt"


def test_publish_handle_does_not_write_through_planted_symlink(tmp_path):
    # Sandboxed code can plant outputs/x.csv -> <host file>. The parent must replace the link,
    # never write through it.
    host = _Host()
    sess = _session(tmp_path, host)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    (sess.root / "outputs").mkdir()
    os.symlink(victim, sess.root / "outputs" / "x.csv")
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    _tool(sess, "publish_handle")(h.id, format="csv", name="x.csv")
    assert victim.read_text() == "precious"


def test_publish_handle_refuses_symlinked_outputs_dir(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, sess.root / "outputs")
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")
    out = _tool(sess, "publish_handle")(h.id, format="csv")
    assert "error" in out and host.calls == []
    assert list(elsewhere.iterdir()) == []


# --- sandbox publish() --------------------------------------------------------------------

_PUBLISH_CODE = (
    "import os\nfrom tether_sandbox import publish\n"
    "os.makedirs('outputs', exist_ok=True)\n"
    "open('outputs/r.csv', 'w').write('a\\n1\\n')\n"
    "publish('outputs/r.csv', name='report.csv', description='desc')\n"
)


def test_sandbox_publish_triggers_callback_after_exit(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    res = sess.sandbox.run_code(_PUBLISH_CODE)
    assert res.error is None, res.error
    assert len(host.calls) == 1
    assert host.calls[0].source == "run_python" and host.calls[0].name == "report.csv"
    assert host.copies["report.csv"] == b"a\n1\n"
    assert res.published[0]["name"] == "report.csv"
    assert res.published[0]["host"] == {"file_id": "F1"}


def test_sandbox_publish_absolute_path_under_root(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    code = _PUBLISH_CODE.replace("publish('outputs/r.csv'", "publish(os.path.abspath('outputs/r.csv')")
    res = sess.sandbox.run_code(code)
    assert res.published and res.published[0]["rel_path"] == "outputs/r.csv"


def test_sandbox_publish_without_callback_raises_clearly(tmp_path):
    sess = _session(tmp_path, None)
    res = sess.sandbox.run_code(_PUBLISH_CODE)
    assert res.exit_code != 0
    assert "publishing is not enabled" in res.stderr
    assert res.published == []


def test_forged_publish_request_is_rejected(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    (tmp_path / "secret.txt").write_text("s")
    code = (
        "import json, os\n"
        "with open(os.path.join(os.environ['TETHER_RUN_DIR'], 'publish.jsonl'), 'a') as f:\n"
        f"    f.write(json.dumps({{'path': {str(tmp_path / 'secret.txt')!r}}}) + '\\n')\n"
        "    f.write(json.dumps({'path': '../secret.txt'}) + '\\n')\n"
        "    f.write(json.dumps({'path': 'handles/_manifest.json'}) + '\\n')\n"
        "    f.write(json.dumps({'path': '.tether/manifest.json'}) + '\\n')\n"
        "    f.write('not json\\n')\n"
        "    f.write(json.dumps({'path': ['x']}) + '\\n')\n"
    )
    res = sess.sandbox.run_code(code)
    assert host.calls == []
    assert len(res.published) == 5           # every well-formed request was considered
    assert all("error" in p for p in res.published)


def test_publish_requests_ignored_when_script_fails(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    res = sess.sandbox.run_code(_PUBLISH_CODE + "raise SystemExit(3)\n")
    assert host.calls == []
    assert "publication" in (res.error or "")


def test_publish_request_count_is_bounded(tmp_path):
    host = _Host()
    sess = _session(tmp_path, host)
    code = _PUBLISH_CODE + "for _ in range(200):\n    publish('outputs/r.csv')\n"
    res = sess.sandbox.run_code(code)
    assert len(host.calls) <= 64
    assert "too many" in (res.error or "")


# --- contextvars + agent loop -------------------------------------------------------------

HOST_RUN = contextvars.ContextVar("HOST_RUN", default=None)


def _agent_run(tmp_path, script, host, setup=None):
    cfg = TetherConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"
    t = Tether(cfg, client=StubChatClient(script), bundles=("code", "deliver"), on_publish=host)

    async def run():
        conv = await t.aopen("s1")
        if setup:
            setup(conv)
        HOST_RUN.set("run-42")
        try:
            resp = await conv.agent.run("go", session=conv.agent_session)
            return resp.text
        finally:
            await t.aclose_sessions()

    return asyncio.run(run())


def test_contextvar_visible_in_publish_file_callback(tmp_path):
    seen = []

    def host(pf):
        seen.append(HOST_RUN.get())
        return {"ok": True}

    def setup(conv):
        _write(conv.session, "outputs/a.csv")

    final = _agent_run(tmp_path, [tool_call("publish_file", {"path": "outputs/a.csv"}),
                                  text("done")], host, setup)
    assert seen == ["run-42"] and final == "done"


def test_contextvar_visible_in_run_python_publish_callback(tmp_path):
    seen = []

    def host(pf):
        seen.append(HOST_RUN.get())
        return {"ok": True}

    _agent_run(tmp_path, [tool_call("run_python", {"code": _PUBLISH_CODE}), text("done")], host)
    assert seen == ["run-42"]


def test_agent_loop_continues_when_host_raises(tmp_path):
    def host(pf):
        raise RuntimeError("nope")

    def setup(conv):
        _write(conv.session, "outputs/a.csv")

    final = _agent_run(tmp_path, [tool_call("publish_file", {"path": "outputs/a.csv"}),
                                  text("recovered")], host, setup)
    assert final == "recovered"


def test_aopen_on_publish_overrides_tether_default(tmp_path):
    calls = []
    cfg = TetherConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"
    t = Tether(cfg, client=StubChatClient([text("x")]), bundles=("deliver",),
               on_publish=lambda pf: calls.append("default"))

    async def run():
        conv = await t.aopen("s1", on_publish=lambda pf: calls.append("override") or {})
        _write(conv.session, "outputs/a.csv")
        conv.session.publish("outputs/a.csv", source="test")
        await t.aclose_sessions()

    asyncio.run(run())
    assert calls == ["override"]


def test_publish_handle_xlsx_without_openpyxl_is_an_error(tmp_path, monkeypatch):
    host = _Host()
    sess = _session(tmp_path, host)
    h = sess.store.put(pd.DataFrame({"a": [1]}), source="t")

    def no_openpyxl(self, *a, **k):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd.DataFrame, "to_excel", no_openpyxl)
    out = _tool(sess, "publish_handle")(h.id, format="xlsx")
    assert "openpyxl" in out["error"] and host.calls == []


def test_sandbox_publish_disabled_without_deliver_bundle(tmp_path):
    host = _Host()
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"), on_publish=host,
                          bundles=("code",))
    res = sess.sandbox.run_code(_PUBLISH_CODE)
    assert "publishing is not enabled" in res.stderr and host.calls == []


def test_sandbox_publish_enabled_with_deliver_bundle(tmp_path):
    host = _Host()
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"), on_publish=host,
                          bundles=("code", "deliver"))
    assert sess.sandbox.run_code(_PUBLISH_CODE).published and len(host.calls) == 1


def test_empty_bundle_selection_keeps_sandbox_publish(tmp_path):
    host = _Host()
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"), on_publish=host, bundles=())
    sess.sandbox.run_code(_PUBLISH_CODE)
    assert len(host.calls) == 1


def test_conversation_without_deliver_disables_sandbox_publish(tmp_path):
    from tether.conversation import Conversation

    async def run():
        conv = await Conversation.acreate(id="c", config=TetherConfig(root_dir=tmp_path / "r"),
                                          client=StubChatClient([text("ok")]),
                                          bundles=("code",), on_publish=_Host())
        try:
            return conv.session.sandbox.publisher
        finally:
            await conv.aclose()

    assert asyncio.run(run()) is None


def test_run_python_description_mentions_publish(tmp_path):
    sess = _session(tmp_path, _Host())
    assert "publish(" in _tool(sess, "run_python").__doc__
