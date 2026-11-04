"""MCP results carry no hidden raw copy for compaction to count (issue #6).

MAF's MCP tools keep a copy of every raw result, up to ``max_host_payload_size_bytes`` (1 MiB
by default), in the result content's ``additional_properties``. The model never sees it, but
the compaction token estimate does, so a result sheath spilled to a short handle summary still
weighed as much as the raw page.
"""
import json
import logging
import sys
from pathlib import Path

import pytest
from agent_framework import MCPStdioTool

from sheath import Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call

_FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_deferred_server.py")
_HOST_PAYLOAD_KEY = "_mcp_tool_result_host_payload"


class _ResultRecorder(StubChatClient):
    """Stub that records every function result it is sent, with its additional_properties."""

    def __init__(self, script):
        super().__init__(script)
        self.results: list = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.results = [c for m in messages for c in m.contents
                        if getattr(c, "type", None) == "function_result"]
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def _mcp(**kwargs) -> MCPStdioTool:
    return MCPStdioTool(name="fixture", command=sys.executable, args=[_FIXTURE], **kwargs)


def _run(tmp_path, tool, script):
    client = _ResultRecorder(script)
    h = Sheath(SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=1024),
               client=client, tools=[tool])
    result = h.solve("go", keep=True)
    return client, result


def test_attaching_sets_the_smallest_host_payload_limit(tmp_path):
    tool = _mcp()
    _run(tmp_path, tool, [tool_call("ping", {}), text("done")])
    assert tool.max_host_payload_size_bytes == 1


def test_a_caller_set_limit_still_warns(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    _run(tmp_path, _mcp(max_host_payload_size_bytes=4096),
         [tool_call("big", {"n": 2000}), text("done")])
    assert any("Omitting MCP Host payload" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_a_caller_set_limit_is_kept(tmp_path):
    tool = _mcp(max_host_payload_size_bytes=4096)
    _run(tmp_path, tool, [tool_call("ping", {}), text("done")])
    assert tool.max_host_payload_size_bytes == 4096


def test_a_spilled_result_weighs_about_as_much_as_its_summary(tmp_path):
    client, result = _run(tmp_path, _mcp(), [tool_call("big", {"n": 2000}), text("done")])
    assert result.handles, "the big page was not spilled"
    [spilled] = client.results
    props = spilled.additional_properties or {}
    assert _HOST_PAYLOAD_KEY not in props
    summary = str(spilled.result)
    serialized = json.dumps(spilled.to_dict(exclude_none=True), default=str)
    # The raw page is ~94 KB; the summary is a few hundred bytes. Allow for the envelope.
    assert len(serialized) < len(summary) * 3 + 2_000


def test_the_omitted_payload_is_not_a_per_call_warning(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    _run(tmp_path, _mcp(), [tool_call("big", {"n": 2000}), tool_call("ping", {}), text("done")])
    omitted = [r for r in caplog.records if "Omitting MCP" in r.getMessage()]
    assert omitted, "no payload was omitted, so this test checks nothing"
    assert all(r.levelno == logging.DEBUG for r in omitted)


@pytest.mark.filterwarnings("ignore:.*MCP progressive disclosure is experimental")
def test_tools_loaded_mid_run_get_the_limit_too(tmp_path):
    tool = _mcp(use_progressive_disclosure=True, always_load=["ping"])
    client, _ = _run(tmp_path, tool, [tool_call("load_tool", {"tool": "big"}),
                                      tool_call("big", {"n": 2000}), text("done")])
    assert all(_HOST_PAYLOAD_KEY not in (c.additional_properties or {}) for c in client.results)
