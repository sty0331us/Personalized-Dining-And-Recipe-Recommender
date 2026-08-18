"""Conversational host: intent routing, MCP-backed lookups, LangGraph recommendations."""

from __future__ import annotations

import json
from typing import Any

from pdr.agents.graph import run_recommendation_workflow
from pdr.agents.nodes import classify_intent
from pdr.mcp.server import get_restaurant_info, get_review, multimodal_search, recommend_by_vibe
from pdr.preference.engine import build_preference_profile
from pdr.ui.formatters import HELP_TEXT, format_recommendations


def _looks_like_vibe(message: str) -> bool:
    vibes = (
        "moody",
        "romantic",
        "zen",
        "cozy",
        "sun-drenched",
        "industrial",
        "glamorous",
        "family",
        "cyberpunk",
    )
    lowered = message.lower()
    return any(word in lowered for word in vibes)


def handle_user_turn(
    message: str,
    user_id: str | None = None,
    uploaded_image: Any | None = None,
) -> str:
    intent = classify_intent(message)
    if intent == "clarification":
        return HELP_TEXT
    if intent == "database":
        return "Use the **Add Restaurant** or **Add Recipe** tabs to update the catalog, then re-run ingest."
    if intent == "lookup" or "review" in message.lower() or "tell me about" in message.lower():
        if _looks_like_vibe(message):
            return f"```json\n{recommend_by_vibe(message)}\n```"
        info = json.loads(get_restaurant_info(message))
        if info.get("status") == "found":
            review = json.loads(get_review(info["results"][0]["name"]))
            return (
                f"**{info['results'][0]['name']}** — {info['results'][0].get('cuisine')} in "
                f"{info['results'][0].get('location')} ({info['results'][0].get('price_range')})\n\n"
                f"{info['results'][0].get('description')}\n\n"
                f"**Your prior review:** {review.get('text') or review.get('message')}"
            )
        if _looks_like_vibe(message):
            return f"```json\n{recommend_by_vibe(message)}\n```"
        return f"```json\n{multimodal_search(message)}\n```"

    result = run_recommendation_workflow(
        user_input=message,
        user_id=user_id,
        intent=intent if intent in {"restaurant", "recipe", "both"} else "both",
        uploaded_image=uploaded_image,
    )
    profile = result.get("user_profile") or build_preference_profile(user_id)
    body = format_recommendations(result.get("final_recommendations") or {})
    summary = profile.get("summary")
    if summary:
        return f"{body}\n\n---\n*Personalization:* {summary}"
    return body
