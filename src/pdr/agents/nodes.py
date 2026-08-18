"""LangGraph node functions for the six specialized recommendation agents."""

from __future__ import annotations

import json
from typing import Any

from pdr.agents.personas import system_prompt
from pdr.agents.state import AgentState
from pdr.data.loaders import catalog
from pdr.llm import LLMNotConfiguredError, invoke_json, invoke_text, llm_is_configured
from pdr.logging_utils import get_logger
from pdr.preference.engine import build_preference_profile

logger = get_logger(__name__)

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from pdr.rag.retriever import MultimodalRetriever

        _retriever = MultimodalRetriever()
    return _retriever


def _call_json(agent_key: str, user_message: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = invoke_json(system_prompt(agent_key), user_message, fallback=fallback)
        return parsed if isinstance(parsed, dict) else fallback
    except LLMNotConfiguredError as exc:
        logger.warning("LLM unavailable for %s: %s", agent_key, exc)
        payload = dict(fallback)
        payload["note"] = str(exc)
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent %s failed: %s", agent_key, exc)
        payload = dict(fallback)
        payload["error"] = str(exc)
        return payload


def classify_intent(user_message: str) -> str:
    prompt = (
        "Classify the user message as ONE of: restaurant, recipe, both, lookup, clarification, database.\n"
        "lookup = asking about a named restaurant or a specific review.\n"
        "Respond with only the label.\n\n"
        f"Message: {user_message}"
    )
    label = ""
    if llm_is_configured():
        try:
            label = invoke_text(system_prompt("intent_router"), prompt).strip().lower()
        except Exception:
            label = ""
    valid = {"restaurant", "recipe", "both", "lookup", "clarification", "database"}
    if label in valid:
        return label
    lowered = user_message.lower()
    if any(word in lowered for word in ("recipe", "cook", "make at home", "how do i make")):
        return "recipe"
    if any(word in lowered for word in ("restaurant", "dine", "eat out", "vibe", "neighborhood")):
        return "restaurant"
    if any(word in lowered for word in ("help", "what can you", "who are you")):
        return "clarification"
    return "both"


def extract_preferences(user_message: str) -> dict[str, Any]:
    fallback = {
        "favorite_cuisines": [],
        "dietary_restrictions": [],
        "dining_occasion": "not specified",
        "price_range": "not specified",
        "flavor_preferences": [],
        "other_preferences": user_message,
    }
    return _call_json(
        "user_profile_generator",
        (
            "Extract preferences as JSON with keys favorite_cuisines, dietary_restrictions, "
            "dining_occasion, price_range, flavor_preferences, other_preferences.\n\n"
            f"Message: {user_message}"
        ),
        fallback,
    )


def node_generate_profile(state: AgentState) -> AgentState:
    logger.info("[Phase 1] Generating user profile")
    historic = build_preference_profile(state.get("user_id"))
    live = extract_preferences(state.get("user_input", ""))
    fallback = {
        "favorite_cuisines": live.get("favorite_cuisines") or [],
        "dietary_restrictions": live.get("dietary_restrictions") or [],
        "dining_occasions": [live.get("dining_occasion", "not specified")],
        "price_range": live.get("price_range", "not specified"),
        "adventurousness_score": 6,
        "flavor_preferences": live.get("flavor_preferences") or [],
        "liked_restaurants": historic.get("liked_restaurants", []),
        "visual_preferences": historic.get("visual_preferences", []),
        "preference_query": historic.get("preference_query", ""),
        "summary": historic.get("summary", ""),
    }
    enriched = _call_json(
        "user_profile_generator",
        (
            "Create a comprehensive dining profile as JSON with keys favorite_cuisines, "
            "dietary_restrictions, dining_occasions, price_range, adventurousness_score, "
            "flavor_preferences, liked_restaurants, visual_preferences, preference_query, summary.\n\n"
            f"Live request: {state.get('user_input', '')}\n"
            f"Historic reviews/photos: {json.dumps(historic, indent=2)}"
        ),
        fallback,
    )
    if not enriched.get("preference_query"):
        enriched["preference_query"] = historic.get("preference_query", "")
    if not enriched.get("liked_restaurants"):
        enriched["liked_restaurants"] = historic.get("liked_restaurants", [])
    return {
        "user_profile": enriched,
        "intent": state.get("intent") or classify_intent(state.get("user_input", "")),
        "workflow_step": "profile_generated",
    }


def _hit_to_restaurant(hit) -> dict[str, Any]:
    restaurants, _, _, _ = catalog()
    match = next(
        (r for r in restaurants if r.item_id == hit.id or r.name.lower() == (hit.name or "").lower()),
        None,
    )
    if match:
        return {
            "id": match.item_id,
            "name": match.name,
            "cuisine": match.cuisine,
            "price": match.price_range,
            "rating": match.rating,
            "location": match.location,
            "vibes": match.vibes,
            "signatures": match.signatures,
            "description": match.description,
            "fused_score": hit.fused,
            "text_score": hit.text_score,
            "img_score": hit.img_score,
            "pref_score": hit.pref_score,
        }
    return {
        "id": hit.id,
        "name": hit.name or hit.id,
        "cuisine": hit.cuisine,
        "price": "$$",
        "rating": None,
        "location": hit.location,
        "description": hit.snippet,
        "fused_score": hit.fused,
        "text_score": hit.text_score,
        "img_score": hit.img_score,
        "pref_score": hit.pref_score,
    }


def _hit_to_recipe(hit) -> dict[str, Any]:
    _, recipes, _, _ = catalog()
    match = next(
        (r for r in recipes if r.recipe_id == hit.id or r.name.lower() == (hit.name or "").lower()),
        None,
    )
    if match:
        return {
            "id": match.recipe_id,
            "name": match.name,
            "cuisine": match.cuisine,
            "difficulty": "Medium" if match.total_time else "Easy",
            "prep_time": match.prep_time,
            "total_time": match.total_time,
            "description": match.image_description,
            "ingredients": match.ingredients[:8],
            "fused_score": hit.fused,
            "text_score": hit.text_score,
            "img_score": hit.img_score,
        }
    return {
        "id": hit.id,
        "name": hit.name or hit.id,
        "cuisine": hit.cuisine,
        "difficulty": "Easy",
        "description": hit.snippet,
        "fused_score": hit.fused,
    }


def node_retrieve_candidates(state: AgentState) -> AgentState:
    logger.info("[Phase 2] Multimodal retrieval + rerank")
    profile = state.get("user_profile") or {}
    query_parts = [
        state.get("user_input", ""),
        " ".join(profile.get("favorite_cuisines") or []),
        " ".join(profile.get("flavor_preferences") or []),
        " ".join(profile.get("liked_restaurants") or []),
        profile.get("preference_query", ""),
    ]
    query = " ".join(part for part in query_parts if part).strip()
    retriever = get_retriever()
    restaurant_hits = retriever.search(
        query=query or "California restaurants",
        preference_query=profile.get("preference_query"),
        image=state.get("uploaded_image"),
    )
    recipe_hits = retriever.search_recipes(
        query=query or "home cooking recipes",
        preference_query=profile.get("preference_query"),
    )
    restaurants = [_hit_to_restaurant(hit) for hit in restaurant_hits if hit.entity_type != "recipe"]
    if not restaurants:
        restaurants = [_hit_to_restaurant(hit) for hit in restaurant_hits]
    recipes = [_hit_to_recipe(hit) for hit in recipe_hits]
    logger.info("Retrieved %s restaurants and %s recipes", len(restaurants), len(recipes))
    return {
        "retrieved_restaurants": restaurants,
        "retrieved_recipes": recipes,
        "fused_hits": [hit.model_dump() for hit in restaurant_hits[:8] + recipe_hits[:8]],
        "workflow_step": "candidates_retrieved",
    }


def node_analyze_trends(state: AgentState) -> AgentState:
    logger.info("[Phase 3a] Analyzing food trends")
    fallback = {
        "trends": [
            {
                "name": "California farm-to-table continuity",
                "description": "Seasonal produce and greenhouse dining remain core to LA dining.",
                "relevance": "Matches Silver Lake / coastal casual retrievals.",
            }
        ]
    }
    analysis = _call_json(
        "food_trend_analyst",
        (
            "Identify 3-5 relevant California dining trends. Return JSON "
            '{"trends": [{"name": str, "description": str, "relevance": str}]}\n\n'
            f"Restaurants: {json.dumps(state.get('retrieved_restaurants', [])[:5], indent=2)}\n"
            f"Recipes: {json.dumps(state.get('retrieved_recipes', [])[:5], indent=2)}"
        ),
        fallback,
    )
    return {"trend_analysis": analysis}


def node_analyze_styles(state: AgentState) -> AgentState:
    logger.info("[Phase 3b] Analyzing food styles")
    fallback = {"cuisines": [], "flavor_profiles": [], "best_style_matches": []}
    analysis = _call_json(
        "food_style_expert",
        (
            "Analyze cuisine types, regional variations, cooking methods, and flavor profiles. "
            'Return JSON {"cuisines": [], "flavor_profiles": [], "best_style_matches": []}\n\n'
            f"User Profile: {json.dumps(state.get('user_profile', {}), indent=2)}\n"
            f"Restaurants: {json.dumps(state.get('retrieved_restaurants', [])[:5], indent=2)}\n"
            f"Recipes: {json.dumps(state.get('retrieved_recipes', [])[:5], indent=2)}"
        ),
        fallback,
    )
    return {"style_analysis": analysis}


def node_evaluate_nutrition(state: AgentState) -> AgentState:
    logger.info("[Phase 3c] Evaluating nutrition")
    fallback = {"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}
    analysis = _call_json(
        "nutrition_expert",
        (
            "Evaluate nutritional fit, allergens, and dietary restrictions. "
            'Return JSON {"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}\n\n'
            f"User Profile: {json.dumps(state.get('user_profile', {}), indent=2)}\n"
            f"Restaurants: {json.dumps(state.get('retrieved_restaurants', [])[:5], indent=2)}\n"
            f"Recipes: {json.dumps(state.get('retrieved_recipes', [])[:5], indent=2)}"
        ),
        fallback,
    )
    return {"nutrition_analysis": analysis}


def node_generate_recommendations(state: AgentState) -> AgentState:
    logger.info("[Phase 4] Generating final recommendations")
    restaurants = state.get("retrieved_restaurants", [])[:5]
    recipes = state.get("retrieved_recipes", [])[:5]
    fallback = {
        "restaurants": [
            {
                "name": item.get("name"),
                "cuisine": item.get("cuisine"),
                "price": item.get("price"),
                "location": item.get("location"),
                "reasoning": (
                    f"Fused multimodal score {float(item.get('fused_score') or 0):.3f} combining "
                    "California restaurant text, dish-photo similarity, and your prior reviews."
                ),
            }
            for item in restaurants
        ],
        "recipes": [
            {
                "name": item.get("name"),
                "cuisine": item.get("cuisine"),
                "difficulty": item.get("difficulty", "Easy"),
                "reasoning": (
                    f"Matches your taste profile and visual dish preferences "
                    f"(fused score {float(item.get('fused_score') or 0):.3f})."
                ),
            }
            for item in recipes
        ],
    }
    recommendations = _call_json(
        "recommendation_expert",
        (
            "Synthesize top restaurant and recipe recommendations. Return JSON "
            '{"restaurants": [{"name": str, "cuisine": str, "price": str, "location": str, "reasoning": str}], '
            '"recipes": [{"name": str, "cuisine": str, "difficulty": str, "reasoning": str}]}. '
            "Each reasoning must cite multimodal evidence (text, photo, or prior review).\n\n"
            f"User Profile: {json.dumps(state.get('user_profile', {}), indent=2)}\n"
            f"Restaurants: {json.dumps(restaurants, indent=2)}\n"
            f"Recipes: {json.dumps(recipes, indent=2)}\n"
            f"Trends: {json.dumps(state.get('trend_analysis', {}), indent=2)}\n"
            f"Styles: {json.dumps(state.get('style_analysis', {}), indent=2)}\n"
            f"Nutrition: {json.dumps(state.get('nutrition_analysis', {}), indent=2)}"
        ),
        fallback,
    )
    intent = state.get("intent", "both")
    if intent == "restaurant":
        recommendations["recipes"] = []
    elif intent == "recipe":
        recommendations["restaurants"] = []
    return {
        "final_recommendations": recommendations,
        "workflow_step": "complete",
    }
