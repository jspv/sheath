"""Watch a parent agent create its own sub-agents at run time and iterate on a poem with them.

One set of instructions goes to a single parent. Under a `SpawnPolicy` the parent spawns a
`timer` sub-agent (it waits 30 seconds, then messages the parent) and a `poet` sub-agent (it
writes a poem about a robot). While the timer runs, the parent critiques each draft and sends the
poet revision instructions. When the timer's message arrives, the parent stops and returns the
best poem so far.

The host decides only what is allowed: which models sub-agents may use (`SpawnPolicy`) and which
host tools may be handed down (`wait_seconds`). The parent decides everything else.

Run it (needs OPENAI_API_KEY in .env, a running podman or docker, and costs a few cents):
    uv run python examples/subagents_poem.py
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from typing import Any

from agent_framework import FunctionMiddleware

from sheath import Sheath, SheathConfig, SpawnPolicy, StatusEvent

PARENT_MODEL = "gpt-6.1-sol"
WORKER_MODEL = "gpt-5-mini"     # the lesser model the sub-agents run on
TIMER_SECONDS = 30
PARENT_NAME = "data-integrator"

_T0 = time.monotonic()


def _say(who: str, text: str) -> None:
    head = f"{time.monotonic() - _T0:6.1f}s  {who:<10}"
    lines = text.splitlines() or [""]
    print(f"{head} {lines[0]}")
    for line in lines[1:]:
        print(f"{'':<18} {line}")


async def wait_seconds(seconds: int) -> str:
    """Wait the given number of seconds, then report that the timer is up."""
    await asyncio.sleep(seconds)
    return f"Timer is up: waited {seconds} seconds."


def show_status(event: StatusEvent) -> None:
    """Sub-agent activity from sheath's status stream, tagged with the sub-agent's path."""
    if event.agent is None and not event.message.startswith("message to "):
        return
    who = event.agent or "parent"
    what = event.message if event.tool == "sheath" else f"{event.tool}: {event.message}"
    _say(who, what)


def _args(context: Any) -> dict:
    args = context.arguments
    return dict(args) if isinstance(args, dict) else args.model_dump()


class ShowDelegation(FunctionMiddleware):
    """The parent's delegation tool calls and what they returned."""

    async def process(self, context: Any, call_next: Any) -> None:
        name = context.function.name.removeprefix("background_agents_")
        args = _args(context)
        if name == "spawn_subagent":
            _say("parent", f"spawn {args['name']} on {args.get('model')}: {args['task']}")
        elif name == "continue_task":
            _say("parent", f"revise task {args['task_id']}:\n" + textwrap.fill(args["text"], 76))
        elif name == "wait_for_first_completion":
            _say("parent", f"wait for first of tasks {args['task_ids']}")
        elif name == "get_task_results":
            _say("parent", f"read result of task {args['task_id']}")
        await call_next()
        result = str(context.result[0].text if isinstance(context.result, list)
                     else context.result)
        if name == "get_task_results":
            _say("result", result)
        elif name in {"spawn_subagent", "continue_task"}:
            _say("", f"-> {result}")


PARENT_INSTRUCTIONS = f"""\
You produce the best possible short poem about a robot, using two sub-agents you create yourself
with spawn_subagent (use model "{WORKER_MODEL}" for both).

1. Spawn `timer` first. Give it tools=["wait_seconds"]. Its instructions: wait {TIMER_SECONDS}
   seconds with wait_seconds, then use send_message to tell "{PARENT_NAME}" (that is you) that
   time is up. Its task: "Start the timer now."
2. Spawn `poet`. Its instructions: reply with the poem only, and with a complete revised poem
   when given critique. Its task: "Write a poem of 8 to 12 lines about a robot."
3. Then loop. Wait for the first completion among the tasks you started.
   - When the poet finishes: read its poem, keep track of the best version so far, critique it
     in two or three sentences, and send concrete enhancement instructions with
     background_agents_continue_task on the poet's task, then wait again.
   - When you receive the timer's message ("[message from timer] ..."), time is up. Stop asking
     for revisions. If the poet is still working, wait for it and consider that last draft too.
   Do not restart or continue the timer.
4. Reply with the best poem to date, followed by one line saying which draft it was and why.
"""


async def main() -> None:
    sheath = Sheath(
        SheathConfig(model=PARENT_MODEL),
        bundles=("files",),
        tools=[wait_seconds],
        spawn=SpawnPolicy(models=(WORKER_MODEL,)),
        on_status=show_status,
        agent_instructions=PARENT_INSTRUCTIONS,
        agent_kwargs={"middleware": [ShowDelegation()]},
    )
    task = "Get me the best robot poem you can in the time the timer allows."
    _say("user", task)
    result = await sheath.asolve(task)
    print()
    if result.error:
        print(f"error: {result.error}")
    print(result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
