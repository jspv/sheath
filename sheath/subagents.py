"""Sub-agent specs and their up-front validation (no agent wiring lives here)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sheath.bundles import selected_bundles
from sheath.config import SheathConfig

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_NAME_PATTERN = "^[A-Za-z][A-Za-z0-9_-]{0,63}$"


@dataclass(frozen=True, kw_only=True)
class SubagentSpec:
    description: str                         # what it is for; shown to the parent model
    instructions: str | None = None          # its domain prompt (agent_instructions)
    bundles: tuple[str, ...] | None = None   # None = the parent's selection
    tools: tuple | list = ()                 # callables, ToolFactory, MCP tools
    model: str | None = None                 # model for a client sheath builds; None = config.model
    client: Any | None = None                # a ready client of its own, instead of model
    max_round_trips: int | None = None       # per-task tool-loop budget


def validate_subagents(
    specs: Mapping[str, SubagentSpec],
    *,
    parent_bundles: tuple[str, ...],
    parent_client: Any,
    client_factory: Callable[[str], Any] | None,
    config: SheathConfig,
) -> dict[str, Any]:
    """Check every spec, then resolve each one's own client: ``{name: client}`` in order.

    All errors are raised before ``client_factory`` is called for any spec.
    """
    if config.subagent_max_concurrent < 1:
        raise ValueError(
            f"subagent_max_concurrent must be >= 1, got {config.subagent_max_concurrent}"
        )
    if config.subagent_task_timeout_s is not None and config.subagent_task_timeout_s <= 0:
        raise ValueError(
            f"subagent_task_timeout_s must be > 0 or None, got {config.subagent_task_timeout_s}"
        )
    parent_allowed = set(selected_bundles(parent_bundles))
    seen_names: dict[str, str] = {}
    seen_clients: list[Any] = []
    for name, spec in specs.items():
        if not isinstance(spec, SubagentSpec):
            raise TypeError(
                f"subagents[{name!r}] must be a SubagentSpec, got {type(spec).__name__}"
            )
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(f"sub-agent name {name!r} must match {_NAME_PATTERN}")
        folded = name.lower()
        if folded in seen_names:
            raise ValueError(
                f"sub-agent names {seen_names[folded]!r} and {name!r} are equal ignoring case"
            )
        seen_names[folded] = name
        if spec.bundles is not None:
            outside = [b for b in selected_bundles(spec.bundles) if b not in parent_allowed]
            if outside:
                raise ValueError(
                    f"sub-agent {name!r} asks for bundles {outside} outside the parent's "
                    f"selection {sorted(parent_allowed)}"
                )
        if spec.max_round_trips is not None and spec.max_round_trips < 1:
            raise ValueError(
                f"sub-agent {name!r}: max_round_trips must be >= 1, got {spec.max_round_trips}"
            )
        if spec.model is not None and spec.client is not None:
            raise ValueError(f"sub-agent {name!r}: set model or client, not both")
        if spec.client is not None:
            if spec.client is parent_client:
                raise ValueError(
                    f"sub-agent {name!r}: client is the parent's client; give it one of its own"
                )
            if any(spec.client is c for c in seen_clients):
                raise ValueError(
                    f"sub-agent {name!r}: client is shared with another sub-agent; "
                    "each needs its own"
                )
            seen_clients.append(spec.client)
        elif client_factory is None:
            raise ValueError(
                f"sub-agent {name!r} has no client: pass client= on the SubagentSpec "
                "or a client_factory"
            )
    resolved: dict[str, Any] = {}
    for name, spec in specs.items():
        resolved[name] = (
            spec.client
            if spec.client is not None
            else client_factory(spec.model or config.model)  # type: ignore[misc]
        )
    return resolved
