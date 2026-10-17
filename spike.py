"""
Spike: prove out a single-Agent harness loop on Microsoft Agent Framework,
running fully locally against Ollama.

Answers two open questions from the research:
  Q1: Can MAF run a long-lived single Agent loop (gather->act->verify) with a
      small set of sharp tools, without authoring a Workflow graph?
  Q2: What does MAF's OpenTelemetry actually emit?

Run: uv run python spike.py
"""

import asyncio

from agent_framework import create_harness_agent
from agent_framework.observability import (
    configure_otel_providers,  # noqa: F401  call with enable_console_exporters=True to dump spans/metrics
    enable_instrumentation,
)
from agent_framework.openai import OpenAIChatClient

# --- a small set of sharp tools (the "give the agent a computer" surface) ---
# Deliberately primitives, not one-tool-per-task. A tiny in-memory "data store"
# stands in for the real data-gathering/integration sources we'll build later.

_STORE = {
    "sales_2025": [120, 95, 140, 0, 210, 0, 175],   # note the zeros (bad rows)
    "sales_2024": [100, 110, 130, 125, 160, 150, 170],
}


def _log(tool: str, **kw) -> None:
    print(f"\n  [tool] {tool}({kw})", flush=True)


def list_datasets() -> list[str]:
    """List the names of datasets available in the store."""
    _log("list_datasets")
    return list(_STORE.keys())


def read_dataset(name: str) -> list[int]:
    """Read the raw integer rows of a dataset by name."""
    _log("read_dataset", name=name)
    return _STORE.get(name, [])


def sum_values(values: list[int]) -> int:
    """Return the arithmetic sum of a list of integers."""
    _log("sum_values", values=values)
    return sum(values)


def verify_no_zero_rows(values: list[int]) -> dict:
    """Verification tool: report whether any rows are zero (treated as invalid)."""
    _log("verify_no_zero_rows", values=values)
    bad = [i for i, v in enumerate(values) if v == 0]
    return {"ok": len(bad) == 0, "zero_row_indices": bad, "n_rows": len(values)}


async def main() -> None:
    # Q2 (already proven in a prior run): instrumentation is on, but skip the
    # verbose console exporters here so the gather->act->verify loop is readable.
    enable_instrumentation()

    # Frontier-ish model via the real OpenAI endpoint (key from .env).
    # gpt-4o-mini: cheap, reliable at multi-step tool calling, non-pro tier.
    client = OpenAIChatClient(model="gpt-4o-mini", env_file_path=".env")

    # Q1: the built-in sheath == the single coherent loop, with todo +
    # compaction + max-iteration cap baked in. No WorkflowBuilder graph.
    agent = create_harness_agent(
        client,
        name="data-integrator",
        agent_instructions=(
            "You integrate small numeric datasets. "
            "IMPORTANT: Work fully autonomously. NEVER stop to ask the user what "
            "to do next -- complete the ENTIRE chain yourself before responding: "
            "(1) list datasets, (2) read the requested one, (3) ALWAYS verify "
            "data quality with verify_no_zero_rows, (4) if it finds invalid "
            "(zero) rows, exclude them, (5) sum the valid rows. "
            "Only produce your final answer AFTER step 5, as a trustworthy total "
            "plus a one-line note on any data issues you handled."
        ),
        tools=[list_datasets, read_dataset, sum_values, verify_no_zero_rows],
        max_context_window_tokens=8192,
        max_output_tokens=1024,
        # Trim the built-in sheath tool surface: a small local model gets
        # distracted by the todo/mode/memory/web-search tools the sheath
        # injects by default. Keep only our sharp data tools + compaction.
        disable_todo=True,
        disable_mode=True,
        disable_memory=True,
        disable_web_search=True,
    )

    task = (
        "What is the total of the sales_2025 dataset? Gather the data, verify "
        "its quality, and give me a trustworthy total with a one-line note on "
        "any data issues you handled."
    )

    print("\n=== TASK ===\n" + task + "\n=== AGENT RUN ===")
    resp = await agent.run(task)
    print("\n=== FINAL ANSWER ===")
    print(resp.text)
    print(f"=== ({len(resp.messages)} messages in the loop) ===")


if __name__ == "__main__":
    asyncio.run(main())
