"""A conversation's workspace can outlive the conversation (issue #7).

For agents whose workspace is the audit trail, closing, ``aclose_sessions`` and idle expiry
must release the session but leave ``<root>/<id>`` for a reaper or the host.
"""
import asyncio

import pytest

from sheath import Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call


def _writer():
    return StubChatClient([tool_call("write_file", {"path": "notes.txt", "content": "audit"}),
                           text("done")])


def _sheath(tmp_path, **cfg) -> Sheath:
    return Sheath(SheathConfig(root_dir=tmp_path / "base", **cfg), client=_writer())


def _converse(h, conv_id="t", **aopen_kwargs):
    async def run():
        conv = await h.aopen(conv_id, **aopen_kwargs)
        await conv.aask("write it down")
        root = conv.session.root
        await h.aclose_sessions()
        return root
    return asyncio.run(run())


def test_with_the_option_set_closing_keeps_the_workspace(tmp_path):
    root = _converse(_sheath(tmp_path, keep_conversation_workspaces=True))
    assert root == tmp_path / "base" / "t"
    assert (root / "notes.txt").read_text() == "audit"


def test_by_default_closing_still_reaps_the_workspace(tmp_path):
    root = _converse(_sheath(tmp_path))
    assert not root.exists()


@pytest.mark.parametrize("config_keep,aopen_keep,kept", [(False, True, True), (True, False, False)])
def test_aopen_keep_overrides_the_config(tmp_path, config_keep, aopen_keep, kept):
    root = _converse(_sheath(tmp_path, keep_conversation_workspaces=config_keep), keep=aopen_keep)
    assert root.exists() is kept


def test_idle_expiry_keeps_the_workspace_too(tmp_path):
    h = _sheath(tmp_path, keep_conversation_workspaces=True, idle_ttl_s=60)

    async def run():
        conv = await h.aopen("t")
        await conv.aask("write it down")
        conv.last_activity -= 1000            # idle past the TTL
        await h.sweep_sessions()
        return conv.session.root

    root = asyncio.run(run())
    assert (root / "notes.txt").read_text() == "audit"


def test_reopening_a_kept_id_resumes_its_files(tmp_path):
    h = _sheath(tmp_path, keep_conversation_workspaces=True)
    _converse(h)

    async def reopen():
        conv = await h.aopen("t")
        try:
            return (conv.session.root / "notes.txt").read_text()
        finally:
            await h.aclose_sessions()

    assert asyncio.run(reopen()) == "audit"


def test_agui_stream_follows_the_config(tmp_path):
    pytest.importorskip("agent_framework_ag_ui")
    h = _sheath(tmp_path, keep_conversation_workspaces=True)

    async def run():
        async for _ in h.agui_stream({"messages": [{"role": "user", "content": "write"}],
                                      "threadId": "t1"}):
            pass
        await h.aclose_sessions()

    asyncio.run(run())
    assert (tmp_path / "base" / "t1" / "notes.txt").read_text() == "audit"
