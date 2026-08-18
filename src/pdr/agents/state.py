"""LangGraph shared state for the hybrid sequential/parallel workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    user_id: str
    intent: str
    uploaded_image: Any
    user_profile: dict[str, Any]
    retrieved_restaurants: list[dict[str, Any]]
    retrieved_recipes: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    trend_analysis: dict[str, Any]
    style_analysis: dict[str, Any]
    nutrition_analysis: dict[str, Any]
    final_recommendations: dict[str, Any]
    workflow_step: str
    errors: list[str]
