"""The tool-loop limits are sheath settings, not a hand-edit of the client (issue #8).

MAF ends a turn after 40 model round-trips and after 3 consecutive tool errors. It reads both
from the chat client's ``function_invocation_configuration``, so that is where sheath applies
them; ``None`` leaves the client's own setting (MAF's defaults on a fresh client).
"""
import asyncio

import pytest

from sheath import Session, Sheath, SheathConfig
from sheath.testing import StubChatClient, text, tool_call


def count(n: int) -> str:
    """Count item n."""
    return f"counted {n}"


_failures = {"left": 0}


def flaky(n: int) -> str:
    """Fail while failures are left, then succeed."""
    if _failures["left"] > 0:
        _failures["left"] -= 1
        raise RuntimeError(f"transient failure on {n}")
    return f"ok {n}"


def _long_script(n_calls: int):
    return [tool_call("count", {"n": i}) for i in range(n_calls)] + [text("FINISHED")]


def _final_text(cfg, client, tools):
    async def go():
        async with Session.create(cfg) as sess:
            agent = await sess.create_agent(client, agent_instructions="x", tools=tools,
                                            bundles=("files",))
            return (await agent.run("go")).text
    return asyncio.run(go())


def test_a_long_run_stops_at_forty_round_trips_by_default(tmp_path):
    client = StubChatClient(_long_script(45))
    assert _final_text(SheathConfig(root_dir=tmp_path / "r"), client, [count]) != "FINISHED"
    assert client.function_invocation_configuration["max_iterations"] == 40


def test_a_raised_round_trip_limit_lets_the_run_finish(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r", tool_loop_max_iterations=50)
    client = StubChatClient(_long_script(45))
    assert _final_text(cfg, client, [count]) == "FINISHED"


def _erroring_script():
    return [tool_call("flaky", {"n": i}) for i in range(6)] + [text("FINISHED")]


def test_five_consecutive_tool_errors_end_the_run_by_default(tmp_path):
    _failures["left"] = 5
    client = StubChatClient(_erroring_script())
    assert _final_text(SheathConfig(root_dir=tmp_path / "r"), client, [flaky]) != "FINISHED"


def test_a_raised_error_limit_survives_five_consecutive_errors(tmp_path):
    _failures["left"] = 5
    cfg = SheathConfig(root_dir=tmp_path / "r", tool_loop_max_consecutive_errors=8)
    client = StubChatClient(_erroring_script())
    assert _final_text(cfg, client, [flaky]) == "FINISHED"


def test_unset_limits_leave_a_hosts_own_client_setting(tmp_path):
    client = StubChatClient(_long_script(45))
    client.function_invocation_configuration["max_iterations"] = 60
    assert _final_text(SheathConfig(root_dir=tmp_path / "r"), client, [count]) == "FINISHED"
    assert client.function_invocation_configuration["max_iterations"] == 60


def test_the_facade_applies_the_limits(tmp_path):
    cfg = SheathConfig(root_dir=tmp_path / "r", tool_loop_max_iterations=50)
    client = StubChatClient(_long_script(45))
    result = Sheath(cfg, client=client, tools=[count]).solve("go")
    assert result.final_text == "FINISHED"


@pytest.mark.parametrize("field,value", [("tool_loop_max_iterations", 0),
                                         ("tool_loop_max_consecutive_errors", -1)])
def test_out_of_range_limits_are_refused(tmp_path, field, value):
    cfg = SheathConfig(root_dir=tmp_path / "r", **{field: value})
    with pytest.raises(ValueError, match=field):
        _final_text(cfg, StubChatClient([text("hi")]), [count])


def test_a_limit_on_a_client_without_a_tool_loop_is_refused(tmp_path):
    class _NoLoop:
        """Implements the chat protocol but has no function-invocation layer."""
        additional_properties: dict = {}

        async def get_response(self, *args, **kwargs):
            raise AssertionError("never called")

    cfg = SheathConfig(root_dir=tmp_path / "r", tool_loop_max_iterations=50)
    with pytest.raises(ValueError, match="function_invocation_configuration"):
        _final_text(cfg, _NoLoop(), [count])
