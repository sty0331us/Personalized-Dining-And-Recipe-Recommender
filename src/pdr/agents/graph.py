"""LangGraph hybrid workflow: sequential profile/RAG, parallel analysis, sequential synthesis."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from pdr.agents.nodes import (
    classify_intent,
    node_analyze_styles,
    node_analyze_trends,
    node_evaluate_nutrition,
    node_generate_profile,
    node_generate_recommendations,
    node_retrieve_candidates,
)
from pdr.agents.state import AgentState
from pdr.config import get_settings
from pdr.logging_utils import get_logger

logger = get_logger(__name__)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("generate_profile", node_generate_profile)
    graph.add_node("retrieve_candidates", node_retrieve_candidates)
    graph.add_node("analyze_trends", node_analyze_trends)
    graph.add_node("analyze_styles", node_analyze_styles)
    graph.add_node("evaluate_nutrition", node_evaluate_nutrition)
    graph.add_node("generate_recommendations", node_generate_recommendations)

    graph.add_edge(START, "generate_profile")
    graph.add_edge("generate_profile", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "analyze_trends")
    graph.add_edge("retrieve_candidates", "analyze_styles")
    graph.add_edge("retrieve_candidates", "evaluate_nutrition")
    graph.add_edge("analyze_trends", "generate_recommendations")
    graph.add_edge("analyze_styles", "generate_recommendations")
    graph.add_edge("evaluate_nutrition", "generate_recommendations")
    graph.add_edge("generate_recommendations", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


def run_recommendation_workflow(
    user_input: str,
    user_id: str | None = None,
    intent: str | None = None,
    uploaded_image: Any | None = None,
) -> AgentState:
    """Execute the four-phase multi-agent recommendation graph."""
    settings = get_settings()
    resolved_intent = intent or classify_intent(user_input)
    initial: AgentState = {
        "user_input": user_input,
        "user_id": user_id or settings.default_user_id,
        "intent": resolved_intent,
        "uploaded_image": uploaded_image,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "fused_hits": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
        "errors": [],
    }
    logger.info("Starting LangGraph workflow intent=%s", resolved_intent)
    return get_compiled_graph().invoke(initial)
