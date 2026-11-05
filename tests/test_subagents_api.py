"""Sub-agents through the public facade: Sheath default, per-call override, client factory."""

from __future__ import annotations

import asyncio

import pytest

from sheath import Sheath, SheathConfig, SubagentSpec
from sheath.session import Session
from sheath.testing import StubChatClient, text


def _cfg(tmp_path) -> SheathConfig:
    cfg = SheathConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"   # skip the dotenv lookup
    return cfg


def _specs(*names, **kw):
    return {n: SubagentSpec(description=f"{n} agent", client=StubChatClient([text("ok")]), **kw)
            for n in names}


def _names(tmp_path, sheath_kw=None, **open_kw):
    async def run():
        t = Sheath(_cfg(tmp_path), client=StubChatClient([text("ok")]), bundles=("code",),
                   **(sheath_kw or {}))
        conv = await t.aopen("s1", **open_kw)
        try:
            return conv.session.subagent_names
        finally:
            await t.aclose_sessions()

    return asyncio.run(run())


def test_sheath_default_used_by_aopen(tmp_path):
    assert _names(tmp_path, {"subagents": _specs("a")}) == ("a",)


def test_aopen_override_replaces_default(tmp_path):
    assert _names(tmp_path, {"subagents": _specs("a")}, subagents=_specs("b")) == ("b",)


def test_empty_dict_override_means_none(tmp_path):
    assert _names(tmp_path, {"subagents": _specs("a")}, subagents={}) == ()


def test_asolve_override(tmp_path, monkeypatch):
    seen = {}
    orig = Session.create_agent

    async def spy(self, client, **kw):
        if "subagents" in kw:       # sub-agents are built through create_agent too
            seen["subagents"] = kw["subagents"]
        return await orig(self, client, **kw)

    monkeypatch.setattr(Session, "create_agent", spy)
    t = Sheath(_cfg(tmp_path), client=StubChatClient([text("ok")]), bundles=("code",),
               subagents=_specs("a"))
    override = _specs("b")
    asyncio.run(t.asolve("hi", subagents=override))
    assert seen["subagents"] == override
    asyncio.run(t.asolve("hi"))
    assert list(seen["subagents"]) == ["a"]


def test_injected_client_without_factory_or_spec_client_raises(tmp_path):
    bare = {"a": SubagentSpec(description="d")}
    with pytest.raises(ValueError):
        _names(tmp_path, {"subagents": bare})


def test_client_factory_called_per_subagent_with_model(tmp_path):
    seen, made = [], []

    def rec(model):
        seen.append(model)
        c = StubChatClient([text("ok")])
        made.append(c)
        return c

    parent = StubChatClient([text("ok")])
    cfg = _cfg(tmp_path)
    subs = {"a": SubagentSpec(description="a", model="m1"), "b": SubagentSpec(description="b")}

    async def run():
        t = Sheath(cfg, client=parent, client_factory=rec, bundles=("code",), subagents=subs)
        conv = await t.aopen("s1")
        await t.aclose_sessions()
        return conv

    asyncio.run(run())
    assert seen == ["m1", cfg.model]
    assert all(c is not parent for c in made)


def test_default_factory_when_no_client_injected(tmp_path, monkeypatch):
    import agent_framework.openai as oa

    calls = []

    def fake(**kw):
        calls.append(kw)
        return StubChatClient([text("ok")])

    monkeypatch.setattr(oa, "OpenAIChatClient", fake)
    subs = {"a": SubagentSpec(description="a", model="m1")}

    async def run():
        t = Sheath(_cfg(tmp_path), bundles=("code",), subagents=subs)
        await t.aopen("s1")
        await t.aclose_sessions()

    asyncio.run(run())
    assert {"model": "m1", "env_file_path": ".env"} in calls


def test_subagentspec_exported():
    import sheath

    assert sheath.SubagentSpec is sheath.subagents.SubagentSpec


def test_sheath_copies_the_subagents_mapping(tmp_path):
    subs = _specs("a")

    async def run():
        t = Sheath(_cfg(tmp_path), client=StubChatClient([text("ok")]), bundles=("code",),
                   subagents=subs)
        subs.update(_specs("b"))   # a later change to the caller's dict must not leak in
        conv = await t.aopen("s1")
        try:
            return conv.session.subagent_names
        finally:
            await t.aclose_sessions()

    assert asyncio.run(run()) == ("a",)
