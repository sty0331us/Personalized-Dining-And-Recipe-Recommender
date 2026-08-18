"""FastMCP server exposing restaurant lookup, vibe search, reviews, and multimodal RAG."""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from pdr.agents.graph import run_recommendation_workflow
from pdr.config import get_settings
from pdr.data.loaders import catalog, load_culinary_map
from pdr.preference.engine import build_preference_profile
from pdr.rag.retriever import MultimodalRetriever

mcp = FastMCP("Connoisseur-Server")


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """Raw California Culinary Map used as the restaurant text corpus."""
    return load_culinary_map()


@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a California restaurant by name and return structured details."""
    restaurants, _, _, _ = catalog()
    query = restaurant_name.lower().strip()
    matches = [
        r.model_dump()
        for r in restaurants
        if query in r.name.lower() or r.name.lower() in query
    ]
    if not matches:
        return _json(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            }
        )
    return _json({"status": "found", "count": len(matches), "results": matches})


@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants matching a vibe keyword, then rerank with multimodal RAG."""
    restaurants, _, _, _ = catalog()
    vibe_lower = vibe.lower().strip()
    structured = []
    for restaurant in restaurants:
        haystack = " ".join([*restaurant.vibes, restaurant.description, restaurant.environment]).lower()
        if vibe_lower in haystack:
            structured.append(
                {
                    "name": restaurant.name,
                    "location": restaurant.location,
                    "cuisine": restaurant.cuisine,
                    "rating": restaurant.rating,
                    "vibes": restaurant.vibes,
                    "price_range": restaurant.price_range,
                }
            )
    retriever = MultimodalRetriever()
    fused = [hit.model_dump() for hit in retriever.search(query=vibe, top_n=6)]
    map_text = load_culinary_map()
    excerpts = [para.strip()[:300] for para in map_text.split("\n\n") if vibe_lower in para.lower()][:5]
    return _json(
        {
            "vibe_searched": vibe,
            "structured_matches": structured[:12],
            "fused_multimodal": fused,
            "raw_text_excerpts": excerpts,
        }
    )


@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the user's review and photo captions for a restaurant."""
    _, _, reviews, _ = catalog()
    query = restaurant_name.lower().strip()
    match = next((r for r in reviews if query in r.restaurant_name.lower()), None)
    if not match:
        return _json(
            {
                "status": "not_found",
                "message": f"No review found for restaurant '{restaurant_name}'.",
            }
        )
    return _json({"status": "found", **match.model_dump()})


@mcp.tool()
def multimodal_search(query: str, top_n: int = 6) -> str:
    """Run multimodal RAG: MiniLM text + CLIP image search with fused reranking."""
    retriever = MultimodalRetriever()
    hits = retriever.search(query=query, top_n=top_n)
    return _json({"query": query, "results": [hit.model_dump() for hit in hits]})


@mcp.tool()
def personalized_recommend(user_message: str, intent: str = "both") -> str:
    """Run the LangGraph multi-agent workflow using the user's review/photo history."""
    settings = get_settings()
    profile = build_preference_profile(settings.default_user_id)
    result = run_recommendation_workflow(
        user_input=user_message,
        user_id=settings.default_user_id,
        intent=intent,
    )
    return _json(
        {
            "preference_summary": profile.get("summary"),
            "recommendations": result.get("final_recommendations", {}),
            "workflow_step": result.get("workflow_step"),
        }
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
