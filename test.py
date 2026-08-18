"""Quick MCP lookup used during development."""

from __future__ import annotations

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_test() -> None:
    server_params = StdioServerParameters(command="python3", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_restaurant_info",
                arguments={"restaurant_name": "Iron"},
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(run_test())
