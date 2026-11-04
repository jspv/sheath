"""SubagentSpec and validate_subagents: the validation table of the sub-agents spec (issue #9)."""

from __future__ import annotations

import pytest

from sheath.bundles import SUBAGENT_INSTRUCTIONS
from sheath.config import SheathConfig
from sheath.subagents import SubagentSpec, validate_subagents
from sheath.testing import StubChatClient

PARENT = StubChatClient([])


def _spec(**kw):
    return SubagentSpec(description="d", **kw)


def _validate(specs, **kw):
    calls: list[str] = []

    def factory(model):
        calls.append(model)
        return StubChatClient([])

    args = dict(
        parent_bundles=("files",),
        parent_client=PARENT,
        client_factory=factory,
        config=SheathConfig(),
    )
    args.update(kw)
    out = validate_subagents(specs, **args)
    return out, calls


def test_spec_has_no_subagents_field():
    with pytest.raises(TypeError):
        SubagentSpec(description="d", subagents={})


def test_non_spec_value_rejected():
    with pytest.raises(TypeError, match="SubagentSpec"):
        _validate({"a": {"description": "d"}})


@pytest.mark.parametrize("name", ["", "1abc", "has space", "a" * 65, "ünï"])
def test_bad_names(name):
    with pytest.raises(ValueError, match=r"\[A-Za-z\]"):
        _validate({name: _spec()})


def test_good_names():
    out, _ = _validate({"a": _spec(), "B-b_9": _spec(), "x" * 64: _spec()})
    assert list(out) == ["a", "B-b_9", "x" * 64]


def test_case_insensitive_duplicate():
    with pytest.raises(ValueError, match="ignoring case"):
        _validate({"Alpha": _spec(), "alpha": _spec()})


def test_bundle_outside_parent():
    with pytest.raises(ValueError, match="web"):
        _validate({"a": _spec(bundles=("web",))})


def test_parent_empty_bundles_means_all():
    out, _ = _validate({"a": _spec(bundles=("web",))}, parent_bundles=())
    assert "a" in out


def test_budget_below_one():
    with pytest.raises(ValueError, match="max_round_trips"):
        _validate({"a": _spec(max_round_trips=0)})


def test_model_and_client_both_set():
    with pytest.raises(ValueError, match="model or client"):
        _validate({"a": _spec(model="m", client=StubChatClient([]))})


def test_parent_client_reused_rejected():
    with pytest.raises(ValueError, match="parent"):
        _validate({"a": _spec(client=PARENT)})


def test_client_shared_across_specs_rejected():
    shared = StubChatClient([])
    with pytest.raises(ValueError, match="shared"):
        _validate({"a": _spec(client=shared), "b": _spec(client=shared)})


def test_no_client_no_factory():
    with pytest.raises(ValueError, match=r"client=.*client_factory"):
        _validate({"a": _spec()}, client_factory=None)


def test_factory_gets_spec_model_else_config_model():
    _, calls = _validate({"a": _spec(model="m-small"), "b": _spec()})
    assert calls == ["m-small", SheathConfig().model]


def test_spec_client_wins_over_factory():
    own = StubChatClient([])
    out, calls = _validate({"a": _spec(client=own)})
    assert out["a"] is own
    assert calls == []


@pytest.mark.parametrize(
    "cfg",
    [SheathConfig(subagent_max_concurrent=0), SheathConfig(subagent_task_timeout_s=0)],
)
def test_config_bounds(cfg):
    with pytest.raises(ValueError, match="subagent_"):
        _validate({"a": _spec()}, config=cfg)


def test_timeout_none_allowed():
    _validate({"a": _spec()}, config=SheathConfig(subagent_task_timeout_s=None))


def test_factory_not_called_when_validation_fails():
    calls = []
    with pytest.raises(ValueError):
        validate_subagents(
            {"a": _spec(), "b": _spec(max_round_trips=0)},
            parent_bundles=("files",),
            parent_client=PARENT,
            client_factory=lambda m: calls.append(m),
            config=SheathConfig(),
        )
    assert calls == []


def test_instructions_constant():
    assert "sub-agent" in SUBAGENT_INSTRUCTIONS
