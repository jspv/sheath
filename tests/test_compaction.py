"""Compaction keeps the original request (issue #5).

The framework's default context-window compaction truncates the oldest non-system groups once
the conversation passes 80% of the input budget. In a one-shot task the oldest user group is
the task itself, so sheath builds the strategy with ``preserve_first_user_group=True``.
"""

import asyncio

from agent_framework import ContextWindowCompactionStrategy

from sheath import Session, Sheath, SheathConfig
from sheath.bundles import CORE_INSTRUCTIONS
from sheath.testing import StubChatClient, text, tool_call

_REQUEST = "ORIGINAL-REQUEST-7f3a: total the EU rows"


class _RecordingClient(StubChatClient):
    """Stub that records the text of every message sent on each model call."""

    def __init__(self, script):
        super().__init__(script)
        self.calls: list[str] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls.append(" ".join(
            str(getattr(c, "result", None) or getattr(c, "text", None) or "")
            for m in messages for c in m.contents))
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def fetch_chunk(n: int) -> str:
    """Return chunk `n` of a long report."""
    return f"chunk {n}: " + "lorem ipsum " * 300   # ~3.6 KB: under the spill threshold


def _long_run(n_calls: int):
    return [tool_call("fetch_chunk", {"n": i}) for i in range(n_calls)] + [text("done")]


def _small_budget(tmp_path, **extra) -> SheathConfig:
    # Input budget 4,000 tokens: tool eviction at 2,000, truncation at 3,200. Each tool result
    # is ~900 tokens, so a 12-call run crosses the truncation threshold several times.
    return SheathConfig(root_dir=tmp_path / "r", max_context_window_tokens=4_500,
                        max_output_tokens=500, **extra)


def _run(cfg, client, **maf_kwargs):
    async def go():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(client, agent_instructions="x", tools=[fetch_chunk],
                                            bundles=("files",), **maf_kwargs)
            await agent.run(_REQUEST)
    asyncio.run(go())


def test_a_long_run_still_sends_the_original_request_on_the_last_call(tmp_path):
    client = _RecordingClient(_long_run(12))
    _run(_small_budget(tmp_path), client)
    assert len(client.calls) == 13
    # The run really was compacted: the first chunk's full text is gone by the end...
    assert "chunk 0: lorem" not in client.calls[-1]
    # ...but the task survived.
    assert _REQUEST in client.calls[-1]


def test_a_caller_strategy_is_used_unchanged(tmp_path):
    seen = []

    class _Spy:
        async def __call__(self, messages):
            seen.append(len(messages))
            return False

    client = _RecordingClient(_long_run(2))
    _run(_small_budget(tmp_path), client, before_compaction_strategy=_Spy())
    assert seen, "the caller's before-strategy was never called"


def test_keep_last_tool_call_groups_reaches_the_strategy(tmp_path):
    built = []
    real = ContextWindowCompactionStrategy.__init__

    def spy(self, **kwargs):
        built.append(kwargs)
        real(self, **kwargs)

    ContextWindowCompactionStrategy.__init__ = spy
    try:
        _run(_small_budget(tmp_path, keep_last_tool_call_groups=2), _RecordingClient([text("hi")]))
    finally:
        ContextWindowCompactionStrategy.__init__ = real
    # The framework also builds its own default whenever the token budgets are set, and then
    # uses the caller's strategy instead; look for sheath's among them.
    assert any(k.get("keep_last_tool_call_groups") == 2
               and k.get("preserve_first_user_group") is True for k in built)


def test_the_facade_path_keeps_the_request_too(tmp_path):
    client = _RecordingClient(_long_run(12))
    h = Sheath(_small_budget(tmp_path), client=client, tools=[fetch_chunk])
    asyncio.run(h.asolve(_REQUEST))
    assert _REQUEST in client.calls[-1]


def test_core_instructions_warn_that_old_tool_results_may_be_collapsed():
    assert "collapsed" in CORE_INSTRUCTIONS
    assert "handle" in CORE_INSTRUCTIONS and "file" in CORE_INSTRUCTIONS
