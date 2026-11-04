import asyncio
import os

import pytest

from sheath import Sheath, SheathConfig, SubagentSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("SHEATH_LIVE") != "1",
    reason="set SHEATH_LIVE=1 (and OPENAI_API_KEY in .env) to run the live sub-agent test",
)


def test_live_delegates_word_count_to_subagent(tmp_path):
    """A real parent writes a file and has the `counter` sub-agent report its word count."""
    tools_called: list[str] = []

    def spy(event):
        if event.tool:
            tools_called.append(event.tool)

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
    assert "9" in result.final_text
    assert "background_agents_start_task" in tools_called
