import httpx

from sheath import SheathConfig, Session
from sheath.spill import make_spill_parser
from sheath.status import StatusBus, bind_bus
from sheath.tools.code import run_python
from sheath.tools.documents import read_document
from sheath.tools.fetch import fetch_url
from sheath.tools.web import web_extract, web_search


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=64))


def _collect():
    bus = StatusBus()
    events = []
    bus.subscribe(events.append)
    return bus, events


def test_read_document_emits_converting_status(tmp_path):
    sess = _session(tmp_path)
    (sess.root / "d.pdf").write_bytes(b"x")
    bus, events = _collect()
    with bind_bus(bus):
        read_document(sess, "d.pdf", convert=lambda src: "# ok")
    assert any(e.tool == "read_document" and "converting" in e.message.lower() for e in events)


def test_run_python_emits_running_status(tmp_path):
    sess = _session(tmp_path)
    bus, events = _collect()
    with bind_bus(bus):
        run_python(sess, code="from sheath_sandbox import emit\nemit(1)\n")
    assert any(e.tool == "run_python" for e in events)


def test_spill_emits_stored_status(tmp_path):
    sess = _session(tmp_path)
    parse = make_spill_parser(sess, "big_tool")
    bus, events = _collect()
    with bind_bus(bus):
        parse({"rows": list(range(500))})              # over the 64-byte threshold -> spills
    stored = [e for e in events if e.tool == "big_tool" and "stored" in e.message.lower()]
    assert stored and "h1" in stored[0].message


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_url_emits_fetching_status(tmp_path):
    sess = _session(tmp_path)
    bus, events = _collect()

    def handler(request):
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})

    with bind_bus(bus):
        fetch_url(sess, "https://example.com/x", client=_mock_client(handler))
    assert any(e.tool == "fetch_url" and "example.com" in e.message for e in events)


def test_web_search_emits_searching_status(tmp_path):
    sess = _session(tmp_path)
    sess.config.search.api_key = "test-key"
    bus, events = _collect()

    def handler(request):
        return httpx.Response(200, json={"answer": None, "results": []})

    with bind_bus(bus):
        web_search(sess, "model pricing", client=_mock_client(handler))
    assert any(e.tool == "web_search" and "pricing" in e.message for e in events)


def test_web_extract_emits_extracting_status(tmp_path):
    sess = _session(tmp_path)
    sess.config.search.api_key = "test-key"
    bus, events = _collect()

    def handler(request):
        return httpx.Response(200, json={"results": [{"raw_content": "body", "url": "https://e/x"}]})

    with bind_bus(bus):
        web_extract(sess, "https://e/x", client=_mock_client(handler))
    assert any(e.tool == "web_extract" and "extracting" in e.message.lower() for e in events)
