"""MCP client smoke test for the Connoisseur tool server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Root

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
PROJECT_DIR = Path(__file__).parent.resolve()

server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])


def list_roots() -> list[Root]:
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]


async def call_tool(tool_name: str, arguments: dict) -> dict:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, list_roots_callback=list_roots) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return json.loads(result.content[0].text)


async def verify_connection() -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, list_roots_callback=list_roots) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            print("Discovered tools:", names)
            for required in (
                "get_restaurant_info",
                "recommend_by_vibe",
                "get_review",
                "multimodal_search",
                "personalized_recommend",
            ):
                assert required in names, f"missing tool {required}"
            print("All required MCP tools verified.")


async def main() -> None:
    await verify_connection()
    print(json.dumps(await call_tool("get_restaurant_info", {"restaurant_name": "Iron & Embers"}), indent=2))
    vibe = await call_tool("recommend_by_vibe", {"vibe": "moody"})
    print("Moody structured matches:", len(vibe.get("structured_matches", [])))


if __name__ == "__main__":
    asyncio.run(main())
