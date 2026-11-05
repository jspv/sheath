# wait_for_message Implementation Plan

**Goal:** let an agent block until another agent messages it, and tell a sender when its message
went to an agent that is not running.

**Problem (seen live):** a sub-agent told to "wait for a message from alpha" has no way to wait:
with no tasks of its own it has no wait tool, so it ends its turn. A message sent after that goes
to its inbox and is seen only if its creator continues it. `send_message` still answered
"Message sent to beta.", so neither the sender nor the creator could tell. `test_spawn_live`
failed 2 of 8 runs this way, on core 1.22 and 1.23 alike.

**Architecture:** a new `Delegation.wait_message` beside `wait_first`, sharing its wake path:
`deliver(..., message=True)` sets every wait registered in `node._waits` for the run it reaches,
and a message queued on the run's session but not yet seen ends the wait at once. The wait runs
inside `AgentNode.waiting()`, so a waiting sub-agent frees its concurrency slot and its clock
pauses. `MessagingProvider` offers the tool wherever it offers `send_message`.

## Decisions

- **Tool:** `wait_for_message(timeout_s: float | None = None) -> str`, offered with
  `send_message` (only under a spawn policy), to the top agent and every sub-agent.
- **Returns:** `MESSAGE_ARRIVED` (the existing string) when a message arrives or is already
  queued unseen on this run; on timeout `"No message arrived within <n> seconds. Call this tool
  again to keep waiting."`; `Error: ` strings when the session is closing or `timeout_s` is not a
  positive finite number (bools refused).
- **Bound:** the wait is at most the run's background-task wait timeout
  (`background_agents_wait_timeout_seconds`, default 300 s); a larger `timeout_s` is clamped to
  it, an omitted one uses it. The clock of a sub-agent's task timeout is paused meanwhile.
- **Wakes on messages only.** Task completion notices do not end it; the tool description points
  to `background_agents_wait_for_first_completion`, which also returns when a message arrives.
- **`send_message` to an agent with no run in progress** answers
  `"Message sent to <path>. <path> is not running: it will see the message when its next task
  starts."` A running recipient keeps `"Message sent to <path>."`.
- **Live test:** the prompt tells beta to use `wait_for_message`, and spawns beta before alpha
  (the old prompt said both "first spawn alpha" and "spawn beta before alpha").

## Tasks (TDD, one commit each)

1. `Delegation.wait_message` + tool + `send_message` idle reply. Tests in
   `tests/test_messaging.py`:
   - a sub-agent blocked in `wait_for_message` is woken by a sibling's message, which is in its
     next model call; its `_waits` is empty afterwards;
   - its slot is free while it waits and every slot is back afterwards;
   - a message queued unseen on the run returns the wait at once;
   - timeout returns the timeout string; an oversized `timeout_s` is clamped to the bound;
   - bad `timeout_s` (0, -1, NaN, inf, True) and a closing session give `Error: `;
   - `send_message` to an idle agent gives the not-running reply; to a running one, unchanged;
   - `wait_for_message` is offered with `send_message`, and not without a policy.
2. Docs: README messaging paragraph, CHANGELOG Unreleased; live test prompt.
