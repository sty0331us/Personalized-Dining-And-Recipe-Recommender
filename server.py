"""Thin launcher kept at the repo root for the FastMCP server."""

from pdr.mcp.server import main, mcp

__all__ = ["mcp", "main"]

if __name__ == "__main__":
    main()
