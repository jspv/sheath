import asyncio
import os

import pytest

from sheath import Sheath, SheathConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("SHEATH_LIVE") != "1",
    reason="set SHEATH_LIVE=1 (and OPENAI_API_KEY in .env) to run the live smoke test",
)


def test_live_gather_act_verify(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r")
    h = Sheath(cfg)

    async def run():
        # solve() is one-shot and builds/reaps its own session, so seed a handle on a
        # persistent Conversation instead, then ask against that same workspace.
        conv = await h.aopen("seed")
        conv.session.store.put({"sales": [120, 0, 210, 0, 95]}, source="seed", id="h1")
        try:
            return await conv.aask(
                "Dataset handle 'h1' holds sales numbers. Total the valid (non-zero) sales, "
                "exclude any zero rows, and report the total with a note on what you excluded."
            )
        finally:
            await conv.aclose()

    result = asyncio.run(run())
    assert "425" in result.final_text
