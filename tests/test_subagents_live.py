import asyncio
import os
import re

import pytest

from sheath import Sheath, SheathConfig, SubagentSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("SHEATH_LIVE") != "1",
    reason="set SHEATH_LIVE=1 (and OPENAI_API_KEY in .env) to run the live sub-agent test",
)


def test_live_delegates_word_count_to_subagent(tmp_path):
    """A real parent writes a file and has the `counter` sub-agent report its word count."""
    events: list = []

    def spy(event):
        events.append(event)

    h = Sheath(
        SheathConfig(root_dir=tmp_path / "r"),
        on_status=spy,
        subagents={"counter": SubagentSpec(description="counts words in a workspace file",
                                           bundles=("files", "code"))},
    )

    async def run():
        conv = await h.aopen("subagents-live")
        try:
            return await conv.aask(
                "Write the sentence 'the quick brown fox jumps over the lazy dog' to words.txt, "
                "then use the `counter` sub-agent to count the words in words.txt and report "
                "the count."
            )
        finally:
            await conv.aclose()

    result = asyncio.run(run())
    assert re.search(r"\b9\b", result.final_text)
    # The sub-agent wrapper emits "started" tagged with the sub-agent's name when a task runs.
    assert any(e.agent == "counter" and e.message == "started" for e in events)
