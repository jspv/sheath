import asyncio

import pytest

from sheath import SheathConfig, Sheath, report_progress
from sheath.api import ToolFactory, expand_tools, tool_factory
from sheath.testing import StubChatClient, text, tool_call


def test_solve_is_ephemeral_by_default(tmp_path):
    client = StubChatClient([text("the answer")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "r"), client=client)
    result = h.solve("go")
    assert result.final_text == "the answer"
    assert not (tmp_path / "r").exists()             # one-shot reaped its workspace


def test_solve_keep_retains_workspace(tmp_path):
    client = StubChatClient([text("the answer")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "k"), client=client)
    result = h.solve("go", keep=True)
    assert (tmp_path / "k").exists()                 # audit trail retained
    assert result.session_dir == tmp_path / "k"


def test_aopen_returns_persistent_conversation(tmp_path):
    import asyncio

    async def run():
        h = Sheath(SheathConfig(root_dir=tmp_path / "base"), client=StubChatClient([text("x")]))
        conv = await h.aopen("t1")
        same = await h.aopen("t1")
        assert conv is same                          # open-or-create via the manager
        await h.aclose_sessions()
        return conv

    asyncio.run(run())


def noisy(n: int) -> str:
    """Emit two progress updates, then return."""
    report_progress("step A", tool="noisy", current=1, total=2)
    report_progress("step B", tool="noisy", current=2, total=2)
    return "ok"


def test_on_status_receives_tool_events_end_to_end(tmp_path):
    events = []
    client = StubChatClient([tool_call("noisy", {"n": 1}), text("done")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "s"), client=client,
                tools=[noisy], on_status=events.append)
    result = h.solve("go")
    assert result.final_text == "done"
    seen = [(e.tool, e.message, e.current, e.total) for e in events]
    assert ("noisy", "step A", 1, 2) in seen
    assert ("noisy", "step B", 2, 2) in seen
    assert [e.seq for e in events] == sorted(e.seq for e in events)   # ordered


def test_on_status_can_be_passed_per_call(tmp_path):
    events = []
    client = StubChatClient([tool_call("noisy", {"n": 1}), text("done")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "s2"), client=client, tools=[noisy])
    h.solve("go", on_status=events.append)                # per-call sets the sink
    assert any(e.message == "step A" for e in events)


def test_raising_on_status_does_not_break_the_run(tmp_path):
    def boom(event):
        raise RuntimeError("subscriber boom")

    client = StubChatClient([tool_call("noisy", {"n": 1}), text("done")])
    h = Sheath(SheathConfig(root_dir=tmp_path / "s3"), client=client,
                tools=[noisy], on_status=boom)
    result = h.solve("go")
    assert result.final_text == "done"                    # status is best-effort; run completes
    assert result.error is None


def _client():
    return StubChatClient([
        tool_call("fetch_big", {"n": 500}),
        text("the answer"),
    ])


def fetch_big(n: int) -> dict:
    """Return a big payload."""
    return {"rows": list(range(n))}


def test_solve_sync_returns_answer_and_spills(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r", spill_threshold_bytes=64)
    h = Sheath(cfg, client=_client(), tools=[fetch_big])
    result = h.solve("go")
    assert result.final_text == "the answer"
    assert result.handles  # developer tool spilled
    assert result.error is None


def test_asolve_matches_solve(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r2", spill_threshold_bytes=64)
    h = Sheath(cfg, client=_client(), tools=[fetch_big])
    result = asyncio.run(h.asolve("go"))
    assert result.final_text == "the answer"
    assert result.handles


class _FailingClient(StubChatClient):
    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        raise RuntimeError("model boom")


def test_solve_preserves_error_on_agent_failure(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "err")
    h = Sheath(cfg, client=_FailingClient([text("unused")]), tools=[])
    result = h.solve("go", keep=True)       # keep retains the work-so-far / audit trail
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert result.final_text == ""
    assert result.session_dir.exists()      # work-so-far / audit trail preserved


def test_tool_factory_wraps_a_callable():
    sentinel = object()
    factory = tool_factory(lambda: sentinel)
    assert isinstance(factory, ToolFactory)
    assert factory.build() is sentinel


def test_expand_tools_calls_each_factory_once():
    calls = []

    def build():
        calls.append(1)
        return f"tool-{len(calls)}"

    tools = [tool_factory(build)]
    first, second = expand_tools(tools), expand_tools(tools)

    assert first == ["tool-1"]
    assert second == ["tool-2"]        # a fresh instance per expansion
    assert len(calls) == 2


def test_expand_tools_passes_plain_tools_through_untouched():
    def my_tool(x: int) -> int:
        """A plain python tool is itself callable; it must not be mistaken for a factory."""
        return x

    out = expand_tools([my_tool])
    assert out == [my_tool]            # not called, not unwrapped


def test_expand_tools_handles_a_mixed_list():
    def my_tool() -> str:
        """Doc."""
        return "direct"

    out = expand_tools([my_tool, tool_factory(lambda: "built")])
    assert out == [my_tool, "built"]


def test_tool_factory_rejects_a_non_callable():
    with pytest.raises(TypeError):
        tool_factory("not callable")


def test_asolve_expands_factories_per_call(tmp_path):
    from sheath.testing import StubChatClient, text

    built = []

    def build():
        built.append(1)
        return lambda: "ok"          # a plain tool the agent could call

    build_tool = tool_factory(build)

    async def run():
        h = Sheath(SheathConfig(root_dir=tmp_path / "r"),
                   client=StubChatClient([text("x")]), tools=[build_tool])
        await h.asolve("go")
        await h.asolve("go")

    asyncio.run(run())
    assert len(built) == 2           # one instance per one-shot conversation


def test_constructor_reads_dotenv_without_exporting_it(tmp_path, monkeypatch):
    import os
    from sheath import Sheath, SheathConfig
    from sheath.testing import StubChatClient, text
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SOME_OTHER_SECRET", raising=False)
    (tmp_path / ".env").write_text("TAVILY_API_KEY=tvly-from-file\nSOME_OTHER_SECRET=s3cret\n")
    cfg = SheathConfig()
    h = Sheath(cfg, client=StubChatClient([text("x")]))
    assert h.config.search.api_key == "tvly-from-file"
    assert cfg.search.api_key is None                  # the caller's object is untouched
    assert "TAVILY_API_KEY" not in os.environ and "SOME_OTHER_SECRET" not in os.environ


def test_environment_key_wins_over_dotenv_and_explicit_key_wins_over_both(tmp_path, monkeypatch):
    from sheath import Sheath, SheathConfig
    from sheath.config import SearchConfig
    from sheath.testing import StubChatClient, text
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TAVILY_API_KEY=from-file\n")
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    client = StubChatClient([text("x")])
    assert Sheath(SheathConfig(), client=client).config.search.api_key == "from-env"
    explicit = SheathConfig(search=SearchConfig(api_key="explicit"))
    assert Sheath(explicit, client=client).config is explicit
