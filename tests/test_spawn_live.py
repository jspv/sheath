"""Live check of runtime sub-agents and messaging (real model, costs money)."""

import asyncio
import os

import pytest

from sheath import Sheath, SheathConfig, SpawnPolicy

pytestmark = pytest.mark.skipif(
    os.environ.get("SHEATH_LIVE") != "1",
    reason="set SHEATH_LIVE=1 (and OPENAI_API_KEY in .env) to run the live spawn test",
)

MODEL = "gpt-5-mini"


def test_live_spawn_two_subagents_one_messages_the_other(tmp_path):
    events: list = []
    h = Sheath(
        SheathConfig(root_dir=tmp_path / "r", model=MODEL),
        bundles=("files",),
        spawn=SpawnPolicy(models=(MODEL,)),
    )

    async def run():
        conv = await h.aopen("spawn-live")
        conv.session.subscribe(events.append)
        try:
            return await conv.aask(
                "Use spawn_subagent to create two sub-agents on model "
                f"{MODEL}. First spawn `alpha` with the task: 'Pick the number 7 and use "
                "send_message to tell `beta` the number, then reply done.' Give alpha "
                "message_to=['beta']. Then spawn `beta` with the task: 'Wait for a message "
                "from alpha containing a number; when it arrives, reply with that number "
                "times 6.' Spawn beta BEFORE alpha so the recipient exists. Wait for both, "
                "then reply with beta's result."
            )
        finally:
            await conv.aclose()

    result = asyncio.run(run())
    names = {e.agent for e in events if e.agent}
    assert {"alpha", "beta"} <= names
    sent = [e for e in events if e.message.startswith("message to ")]
    assert sent and all(e.agent in {"alpha", "beta"} for e in sent)
    assert "42" in result.final_text
