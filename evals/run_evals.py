"""Self-eval sheath: run the data-integration sheath on 10 diverse find-and-analyze
tasks, record the tool-call trace + final answer + any error per task, and write a JSON
result file each task completes so progress can be inspected live.

Run: uv run python evals/run_evals.py [task_id ...]   (no args = all)
Requires OPENAI_API_KEY in .env (uses gpt-5-mini).
"""

import json
import sys
import traceback
from pathlib import Path

from sheath import SandboxConfig, Sheath, SheathConfig

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = Path(__file__).resolve().parent / "runs"
MODEL = "gpt-5-mini"


def _short(value, limit=400):
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + f"… ({len(text)} chars)"


# Each task: id, prompt, optional seed(session) to pre-load handles.
TASKS = [
    {
        "id": "t01_github_json",
        "prompt": ("Fetch https://api.github.com/repos/python/cpython and report the star "
                   "count (stargazers_count), number of open issues (open_issues_count), "
                   "and the primary language."),
    },
    {
        "id": "t02_weather_json",
        "prompt": ("Fetch current London weather from https://wttr.in/London?format=j1 and "
                   "tell me the temperature in Celsius and the weather description right now."),
    },
    {
        "id": "t03_compute_fib",
        "prompt": "Compute the 15th through 20th Fibonacci numbers (F(1)=1) and their sum.",
    },
    {
        "id": "t04_data_integration",
        "prompt": ("Two handles are loaded: 'sales' (monthly sales) and 'targets' (monthly "
                   "targets), same 6 months in order. Tell me which months beat their target "
                   "and the total amount by which sales exceeded target across those months."),
        "seed": lambda s: (
            s.store.put([100, 120, 90, 150, 80, 200], source="seed", id="sales"),
            s.store.put([110, 100, 100, 140, 120, 150], source="seed", id="targets"),
        ),
    },
    {
        "id": "t05_csv_analysis",
        "prompt": ("Here is CSV data:\nname,score\nA,80\nB,95\nC,0\nD,88\nE,0\nF,72\n"
                   "Compute the average score excluding zero (invalid) rows, say how many "
                   "rows you excluded, and name the top scorer."),
    },
    {
        "id": "t06_readme_fetch",
        "prompt": ("Fetch https://raw.githubusercontent.com/psf/requests/main/README.md and "
                   "summarize in 3 bullets what the library is and its key selling points."),
    },
    {
        "id": "t07_openai_pricing",
        "prompt": "What are the current OpenAI API prices for their flagship model (input/output per million tokens)?",
    },
    {
        "id": "t08_current_events",
        "prompt": "Who is the current CEO of OpenAI, and what major product or model did they most recently release? Verify against a current source.",
    },
    {
        "id": "t09_multi_fetch_compare",
        "prompt": ("Compare the star counts of these two repos by fetching them: "
                   "https://api.github.com/repos/microsoft/vscode and "
                   "https://api.github.com/repos/microsoft/TypeScript . "
                   "Which has more stars and by how many?"),
    },
    {
        "id": "t10_transform_and_write",
        "prompt": ("Create a CSV of the first 10 squares with header 'n,n_squared' and write "
                   "it to squares.csv, then tell me the sum of the squares."),
    },
]


def run_task(task: dict) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    # Evals measure model behavior and should not require a container runtime.
    cfg = SheathConfig(root_dir=RUNS / task["id"], model=MODEL,
                       sandbox=SandboxConfig(backend="local"))
    h = Sheath(cfg)
    if task.get("seed"):
        task["seed"](h.session)

    trace = []
    h.solve_problem = task["prompt"]

    def on_tool_call(name, kwargs, result):
        trace.append({"tool": name,
                      "args": {k: _short(v, 120) for k, v in kwargs.items()},
                      "result": _short(result)})

    record = {"id": task["id"], "prompt": task["prompt"]}
    try:
        result = h.solve(task["prompt"], on_tool_call=on_tool_call)
        record.update({
            "final_text": result.final_text,
            "error": result.error,
            "n_tool_calls": len(trace),
            "tools_used": sorted({t["tool"] for t in trace}),
            "trace": trace,
            "handles": list(result.handles),
            "files": result.files,
        })
    except Exception as e:  # noqa: BLE001
        record.update({"final_text": "", "error": f"SHEATH CRASH: {type(e).__name__}: {e}",
                       "n_tool_calls": len(trace), "trace": trace,
                       "traceback": traceback.format_exc()})
    return record


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]
    for task in tasks:
        print(f"=== running {task['id']} ===", flush=True)
        record = run_task(task)
        (RESULTS / f"{task['id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        status = "ERROR" if record.get("error") else "ok"
        print(f"  {task['id']}: {status} | tools={record.get('tools_used')} | "
              f"calls={record.get('n_tool_calls')}", flush=True)
        print(f"  answer: {_short(record.get('final_text', ''), 200)}", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
