"""TETHER-3: domain agent instructions reach the agent through Sheath / aopen / asolve."""

import asyncio

from sheath import Sheath, SheathConfig
from sheath.conversation import Conversation
from sheath.testing import StubChatClient, text


def _instructions(agent) -> str:
    return agent.default_options["instructions"]


def _sheath(tmp_path, **kw) -> Sheath:
    cfg = SheathConfig(root_dir=tmp_path / "base")
    cfg.search.api_key = "x"   # skip the dotenv lookup
    return Sheath(cfg, client=StubChatClient([text("ok")]), bundles=("code",), **kw)


def test_conversation_acreate_passes_agent_instructions(tmp_path):
    async def run():
        conv = await Conversation.acreate(
            id="c", config=SheathConfig(root_dir=tmp_path / "r"),
            client=StubChatClient([text("ok")]), bundles=("code",),
            agent_instructions="DOMAIN-X")
        try:
            return _instructions(conv.agent)
        finally:
            await conv.aclose()

    instr = asyncio.run(run())
    assert instr.endswith("DOMAIN-X")
    assert "run_python" in instr          # sheath's operating manual is still layered first


def test_aopen_agent_instructions(tmp_path):
    async def run():
        t = _sheath(tmp_path)
        conv = await t.aopen("s1", agent_instructions="X-AOPEN")
        try:
            return _instructions(conv.agent)
        finally:
            await t.aclose_sessions()

    assert "X-AOPEN" in asyncio.run(run())


def test_sheath_default_agent_instructions(tmp_path):
    async def run():
        t = _sheath(tmp_path, agent_instructions="X-DEFAULT")
        conv = await t.aopen("s1")
        try:
            return _instructions(conv.agent)
        finally:
            await t.aclose_sessions()

    assert "X-DEFAULT" in asyncio.run(run())


def test_aopen_value_wins_over_sheath_default(tmp_path):
    async def run():
        t = _sheath(tmp_path, agent_instructions="X-DEFAULT")
        conv = await t.aopen("s1", agent_instructions="X-OVERRIDE")
        try:
            return _instructions(conv.agent)
        finally:
            await t.aclose_sessions()

    instr = asyncio.run(run())
    assert "X-OVERRIDE" in instr
    assert "X-DEFAULT" not in instr


def test_asolve_passes_agent_instructions(tmp_path, monkeypatch):
    seen = {}
    real = Conversation.acreate.__func__

    async def spy(cls, **kw):
        conv = await real(cls, **kw)
        seen["instr"] = _instructions(conv.agent)
        return conv

    monkeypatch.setattr(Conversation, "acreate", classmethod(spy))
    t = _sheath(tmp_path, agent_instructions="X-DEFAULT")
    t.solve("go", agent_instructions="X-SOLVE")
    assert "X-SOLVE" in seen["instr"] and "X-DEFAULT" not in seen["instr"]
    t.solve("go")
    assert "X-DEFAULT" in seen["instr"]
