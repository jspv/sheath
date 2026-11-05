"""Runtime sub-agents over AG-UI: tool-call events, live status with paths, cross-request tasks."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("agent_framework_ag_ui")

from sheath import Sheath, SheathConfig, SpawnPolicy  # noqa: E402
from sheath.testing import StubChatClient, text, tool_call  # noqa: E402

POLICY = SpawnPolicy(models=("m1",))


def spawn(name, i, **kw):
    args = {"name": name, "instructions": "You help.", "task": "go", **kw}
    return tool_call("spawn_subagent", args, f"sp{i}")


def wait(ids, i):
    return tool_call("background_agents_wait_for_first_completion", {"task_ids": ids}, f"w{i}")


def results(tid, i):
    return tool_call("background_agents_get_task_results", {"task_id": tid}, f"r{i}")


def cont(tid, value, i):
    return tool_call("background_agents_continue_task", {"task_id": tid, "text": value}, f"c{i}")


def _cfg(tmp_path) -> SheathConfig:
    cfg = SheathConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"
    return cfg


def _dump(ev) -> dict:
    return ev.model_dump(exclude_none=True) if hasattr(ev, "model_dump") else {}


def _type(d) -> str:
    t = d.get("type")
    return getattr(t, "value", t)


def _status(events):
    return [d["value"] for d in events if _type(d) == "CUSTOM" and d.get("name") == "sheath.status"]


async def _collect(sheath, input_data):
    return [_dump(ev) async for ev in sheath.agui_stream(input_data)]


def _req(run, messages):
    return {"threadId": "t1", "runId": run, "messages": messages}


def _user(i, content):
    return {"id": f"u{i}", "role": "user", "content": content}


def _sheath(tmp_path, parent, factory):
    return Sheath(_cfg(tmp_path), client=parent, bundles=("files",), spawn=POLICY,
                  client_factory=factory)


def test_spawn_subagent_streams_as_tool_call(tmp_path):
    async def main():
        parent = StubChatClient([spawn("lead", 1), wait([1], 1), text("final")])
        sheath = _sheath(tmp_path, parent, lambda model: StubChatClient([text("report")]))
        try:
            return await _collect(sheath, _req("r1", [_user(1, "go")]))
        finally:
            await sheath.aclose_sessions()

    events = asyncio.run(main())
    starts = [d for d in events if _type(d) == "TOOL_CALL_START"]
    assert any(d.get("tool_call_name") == "spawn_subagent" and d.get("tool_call_id") == "sp1"
               for d in starts)
    result = next(d for d in events
                  if _type(d) == "TOOL_CALL_RESULT" and d.get("tool_call_id") == "sp1")
    assert "started on agent 'lead'" in result["content"]


def test_spawned_subagent_status_streams_with_its_path(tmp_path):
    async def main():
        parent = StubChatClient([spawn("lead", 1), wait([1], 1), text("final")])
        sheath = _sheath(tmp_path, parent, lambda model: StubChatClient([text("report")]))
        try:
            return await _collect(sheath, _req("r1", [_user(1, "go")]))
        finally:
            await sheath.aclose_sessions()

    status = _status(asyncio.run(main()))
    mine = [s for s in status if s.get("agent") == "lead"]
    assert [s["message"] for s in mine if s["tool"] == "sheath"][:1] == ["started"]
    assert any(s["message"] == "finished" for s in mine)


def test_message_status_carries_sender_path_and_no_text(tmp_path):
    async def main():
        sub = [tool_call("send_message", {"to": "data-integrator", "text": "SECRET-BODY"}, "m1"),
               text("report")]
        parent = StubChatClient([spawn("lead", 1), wait([1], 1), text("final")])
        sheath = _sheath(tmp_path, parent, lambda model: StubChatClient(sub))
        try:
            return await _collect(sheath, _req("r1", [_user(1, "go")]))
        finally:
            await sheath.aclose_sessions()

    events = asyncio.run(main())
    sent = [s for s in _status(events) if s["message"].startswith("message to ")]
    assert sent and sent[0]["agent"] == "lead"
    assert "SECRET-BODY" not in str(sent)


def test_continue_task_across_agui_requests_for_a_spawned_subagent(tmp_path):
    async def main():
        seen = {}

        def record(messages):
            for m in messages:
                for c in m.contents:
                    if c.type == "function_result":
                        seen[c.call_id] = str(c.result)
            return text("done")

        parent = StubChatClient([spawn("lead", 1), wait([1], 1), results(1, 1), text("r1-done"),
                                 cont(1, "more", 1), wait([1], 2), results(1, 2), record])
        sheath = _sheath(tmp_path, parent,
                         lambda model: StubChatClient([text("first"), text("second")]))
        try:
            await _collect(sheath, _req("r1", [_user(1, "go")]))
            await _collect(sheath, _req("r2", [
                _user(1, "go"), {"id": "a1", "role": "assistant", "content": "r1-done"},
                _user(2, "more")]))
        finally:
            await sheath.aclose_sessions()
        return seen

    seen = asyncio.run(main())
    assert "continued with new input" in seen["c1"]
    assert "second" in seen["r2"]
