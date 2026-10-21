"""agent_kwargs: hosts forward create_harness_agent options through the facade paths."""

import asyncio

import pytest
from agent_framework import InlineSkill, SkillFrontmatter, SkillsProvider

from sheath import Session, Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call


class _RecordingClient(StubChatClient):
    """Stub that records the tool names, chat options and messages of every model call."""

    def __init__(self, script):
        super().__init__(script)
        self.tool_names: list[set[str]] = []
        self.options_seen: list[dict] = []
        self.messages_seen: list[str] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.tool_names.append({getattr(t, "name", None) for t in options.get("tools") or []})
        self.options_seen.append(dict(options))
        self.messages_seen.append(" ".join(
            str(getattr(c, "result", None) or getattr(c, "text", None) or "")
            for m in messages for c in m.contents))
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def _skills(body: str) -> SkillsProvider:
    skill = InlineSkill(frontmatter=SkillFrontmatter(name="unit-rules", description="Unit rules."),
                        instructions=body)
    return SkillsProvider([skill], disable_load_skill_approval=True)


def _load_skill_script():
    return [tool_call("load_skill", {"skill_name": "unit-rules"}), text("done")]


def test_sheath_default_agent_kwargs_reach_the_agent(tmp_path):
    client = _RecordingClient(_load_skill_script())
    h = Sheath(SheathConfig(root_dir=str(tmp_path)), client=client,
               agent_kwargs={"skills_provider": _skills("ALWAYS-USE-SI-UNITS")})

    async def run():
        conv = await h.aopen("t1")
        await conv.aask("go")
        await h.aclose_sessions()

    asyncio.run(run())
    assert "load_skill" in client.tool_names[0]
    assert "ALWAYS-USE-SI-UNITS" in client.messages_seen[-1]   # the skill body came back


def test_aopen_agent_kwargs_win_over_the_sheath_default(tmp_path):
    client = _RecordingClient(_load_skill_script())
    h = Sheath(SheathConfig(root_dir=str(tmp_path)), client=client,
               agent_kwargs={"skills_provider": _skills("DEFAULT-BODY")})

    async def run():
        conv = await h.aopen("t1", agent_kwargs={"skills_provider": _skills("PER-CONV-BODY")})
        await conv.aask("go")
        await h.aclose_sessions()

    asyncio.run(run())
    assert "PER-CONV-BODY" in client.messages_seen[-1]
    assert "DEFAULT-BODY" not in client.messages_seen[-1]


def test_asolve_forwards_agent_kwargs(tmp_path):
    client = _RecordingClient(_load_skill_script())
    h = Sheath(SheathConfig(root_dir=str(tmp_path / "r")), client=client)
    asyncio.run(h.asolve("go", agent_kwargs={"skills_provider": _skills("ONE-SHOT-BODY")}))
    assert "ONE-SHOT-BODY" in client.messages_seen[-1]


def test_default_options_merge_across_levels_and_keep_store_off(tmp_path):
    client = _RecordingClient([text("hi")])
    h = Sheath(SheathConfig(root_dir=str(tmp_path)), client=client,
               agent_kwargs={"default_options": {"temperature": 0.2}})

    async def run():
        conv = await h.aopen("t1", agent_kwargs={"default_options": {"top_p": 0.5}})
        await conv.aask("go")
        await h.aclose_sessions()

    asyncio.run(run())
    opts = client.options_seen[0]
    assert (opts.get("temperature"), opts.get("top_p"), opts.get("store")) == (0.2, 0.5, False)


def test_fixed_provider_switches_still_raise(tmp_path):
    h = Sheath(SheathConfig(root_dir=str(tmp_path)), client=StubChatClient([text("hi")]),
               agent_kwargs={"disable_todo": False})
    with pytest.raises(TypeError):
        asyncio.run(h.asolve("go"))


@pytest.mark.parametrize("key", ["shell_executor", "file_access_store", "background_agents"])
def test_unsandboxed_capabilities_are_refused(tmp_path, key):
    # Each would hand the model a path around the sandbox, safe_path or egress.
    async def run():
        async with Session.create(SheathConfig(root_dir=str(tmp_path / "r"))) as sess:
            await sess.create_agent(StubChatClient([text("hi")]), tools=[], bundles=("files",),
                                    **{key: object()})

    with pytest.raises(TypeError, match=rf"{key}.*sandbox"):
        asyncio.run(run())


def test_unsandboxed_capabilities_are_refused_on_the_facade(tmp_path):
    h = Sheath(SheathConfig(root_dir=str(tmp_path)), client=StubChatClient([text("hi")]),
               agent_kwargs={"shell_executor": object()})
    with pytest.raises(TypeError, match="shell_executor.*sandbox"):
        asyncio.run(h.asolve("go"))
