from pdr.agents.graph import build_graph
from pdr.agents.nodes import classify_intent
from pdr.ui.formatters import format_recommendations


def test_langgraph_contains_parallel_analysis_nodes():
    compiled = build_graph()
    graph = compiled.get_graph()
    node_ids = set(graph.nodes)
    for required in {
        "generate_profile",
        "retrieve_candidates",
        "analyze_trends",
        "analyze_styles",
        "evaluate_nutrition",
        "generate_recommendations",
    }:
        assert required in node_ids


def test_keyword_intent_fallback():
    assert classify_intent("How do I make lasagna?") == "recipe"
    assert classify_intent("What can you help me with?") == "clarification"


def test_format_recommendations():
    text = format_recommendations(
        {
            "restaurants": [
                {
                    "name": "Iron & Embers",
                    "cuisine": "Steakhouse",
                    "price": "$$$$",
                    "location": "DTLA",
                    "reasoning": "Moody industrial match.",
                }
            ],
            "recipes": [],
        }
    )
    assert "Iron & Embers" in text
    assert "Restaurant recommendations" in text
