"""A tiny self-contained stdio MCP server for the CopilotKit example.

It exposes one tool, ``sales_rows(region)``, that returns a full year of daily
rows — deliberately large and structured so the sheath's spill middleware turns
the return into a typed handle (instead of flooding the model's context). The
agent then writes ``run_python`` to load that handle and compute the answer.

Run standalone (the example backend launches it for you over stdio):
    uv run python examples/copilotkit/backend/mcp_server.py
"""

from __future__ import annotations

from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sales")

# A few regions with distinct, deterministic profiles. No randomness: the data is
# a pure function of (region, day) so every run is reproducible.
_REGIONS = {"EU": (820.0, 6.1), "NA": (1140.0, 5.4), "APAC": (610.0, 7.3)}


@mcp.tool()
def sales_rows(region: str) -> list[dict]:
    """Return raw daily sales rows for a region across 2025.

    Each row: {date, region, units, revenue, valid}. Some rows are flagged
    ``valid=False`` (returns/cancellations) and should be excluded from totals.
    """
    region = region.upper()
    if region not in _REGIONS:
        raise ValueError(f"unknown region {region!r}; choose from {sorted(_REGIONS)}")

    base_rev, base_units = _REGIONS[region]
    rows: list[dict] = []
    day = date(2025, 1, 1)
    end = date(2025, 12, 31)
    i = 0
    while day <= end:
        # Deterministic seasonal-ish wobble from the day index.
        wobble = ((i * 37) % 100) / 100.0
        units = round(base_units * (0.7 + wobble))
        revenue = round(base_rev * (0.8 + 0.4 * wobble) * units / base_units, 2)
        valid = (i % 17) != 0  # ~6% of rows are returns/cancellations -> invalid
        rows.append({
            "date": day.isoformat(),
            "region": region,
            "units": units,
            "revenue": revenue,
            "valid": valid,
        })
        day += timedelta(days=1)
        i += 1
    return rows


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
