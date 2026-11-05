"""spawn= through the public facade: Sheath default, per-call override, client factory."""

from __future__ import annotations

import asyncio

from sheath import Sheath, SheathConfig, SpawnPolicy
from sheath.session import Session
from sheath.testing import StubChatClient, text


def _cfg(tmp_path) -> SheathConfig:
    cfg = SheathConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"
    return cfg


def _policy(model="m1", **kw):
    return SpawnPolicy(models=(model,), **kw)


def _factory(model):
    return StubChatClient([text("ok")])


def _spy(monkeypatch):
    seen = []
    orig = Session.create_agent

    async def spy(self, client, **kw):
        if "spawn" in kw:
            seen.append(kw["spawn"])
        return await orig(self, client, **kw)

    monkeypatch.setattr(Session, "create_agent", spy)
    return seen


def _sheath(tmp_path, **kw):
    return Sheath(_cfg(tmp_path), client=StubChatClient([text("ok")]), bundles=("code",),
                  client_factory=_factory, **kw)


def test_sheath_default_reaches_aopen(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    pol = _policy()

    async def run():
        t = _sheath(tmp_path, spawn=pol)
        await t.aopen("s1")
        await t.aclose_sessions()

    asyncio.run(run())
    assert seen == [pol]


def test_aopen_override_replaces_default(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    over = _policy("m2")

    async def run():
        t = _sheath(tmp_path, spawn=_policy())
        await t.aopen("s1", spawn=over)
        await t.aclose_sessions()

    asyncio.run(run())
    assert seen == [over]


def test_asolve_override(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    default, over = _policy(), _policy("m2")
    t = _sheath(tmp_path, spawn=default)
    asyncio.run(t.asolve("hi", spawn=over))
    asyncio.run(t.asolve("hi"))
    assert seen == [over, default]


def test_no_spawn_by_default(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    asyncio.run(_sheath(tmp_path).asolve("hi"))
    assert seen == []


def test_default_factory_builds_spawned_client_for_model(tmp_path, monkeypatch):
    import agent_framework.openai as oa

    calls = []

    def fake(**kw):
        calls.append(kw)
        return StubChatClient([text("ok")])

    monkeypatch.setattr(oa, "OpenAIChatClient", fake)

    async def run():
        t = Sheath(_cfg(tmp_path), bundles=("code",), spawn=_policy("m1"))
        conv = await t.aopen("s1")
        try:
            d = conv.session.delegation
            return d.client_factory("m1")
        finally:
            await t.aclose_sessions()

    asyncio.run(run())
    assert {"model": "m1", "env_file_path": ".env"} in calls


def test_spawnpolicy_exported():
    import sheath

    assert sheath.SpawnPolicy is sheath.subagents.SpawnPolicy
    assert "SpawnPolicy" in sheath.__all__


def test_sync_solve_passes_spawn(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    default, over = _policy(), _policy("m2")
    t = _sheath(tmp_path, spawn=default)
    t.solve("hi", spawn=over)
    t.solve("hi")
    assert seen == [over, default]


def test_module_solve_passes_spawn(tmp_path, monkeypatch):
    from sheath import solve

    seen = _spy(monkeypatch)
    pol = _policy()
    solve("hi", config=_cfg(tmp_path), client=StubChatClient([text("ok")]),
          spawn=pol, client_factory=_factory)
    assert seen == [pol]


def test_agui_stream_uses_sheath_default(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("agent_framework_ag_ui")
    seen = _spy(monkeypatch)
    pol = _policy()

    async def run():
        t = _sheath(tmp_path, spawn=pol)
        try:
            async for _ in t.agui_stream(
                    {"threadId": "t1", "runId": "r1",
                     "messages": [{"id": "u1", "role": "user", "content": "go"}]}):
                pass
        finally:
            await t.aclose_sessions()

    asyncio.run(run())
    assert seen == [pol]
