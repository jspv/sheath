"""A tiny FastMCP server for progressive disclosure: one core tool, two loaded on demand."""
import anyio
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("deferredfixture")


@mcp.tool()
def ping() -> str:
    """A small core tool, visible from the start."""
    return "pong"


@mcp.tool()
def big(n: int) -> str:
    """Return a large page of rows."""
    return "\n".join(f"row-{i:05d},{'x' * 40}" for i in range(n))


@mcp.tool()
async def slow(n: int, ctx: Context) -> str:
    """Process n items, reporting progress as it goes."""
    for i in range(n):
        await ctx.report_progress(i + 1, n, f"step {i + 1}")
        await anyio.sleep(0.02)
    return f"done: {n} items"


if __name__ == "__main__":
    mcp.run()
