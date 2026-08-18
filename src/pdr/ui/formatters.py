"""Chat formatting helpers for Gradio."""

from __future__ import annotations

from typing import Any


def format_recommendations(recommendations: dict[str, Any]) -> str:
    output = []
    restaurants = recommendations.get("restaurants") or []
    recipes = recommendations.get("recipes") or []
    if restaurants:
        output.append("### Restaurant recommendations")
        for i, restaurant in enumerate(restaurants, 1):
            output.append(
                f"**{i}. {restaurant.get('name', 'Unknown')}**  \n"
                f"- Cuisine: {restaurant.get('cuisine', 'N/A')}  \n"
                f"- Price: {restaurant.get('price', 'N/A')}  \n"
                f"- Location: {restaurant.get('location', 'N/A')}  \n"
                f"- Why: {restaurant.get('reasoning', '')}"
            )
    if recipes:
        output.append("### Recipe recommendations")
        for i, recipe in enumerate(recipes, 1):
            output.append(
                f"**{i}. {recipe.get('name', 'Unknown')}**  \n"
                f"- Cuisine: {recipe.get('cuisine', 'N/A')}  \n"
                f"- Difficulty: {recipe.get('difficulty', 'N/A')}  \n"
                f"- Why: {recipe.get('reasoning', '')}"
            )
    if not output:
        return "I could not generate recommendations. Add more detail about vibe, cuisine, or dietary needs."
    return "\n\n".join(output)


HELP_TEXT = """I am **Connoisseur Companion**, a California dining and recipe assistant.

I can help with:

- **Restaurant recommendations** grounded in California restaurant copy and dish photos
- **Recipe recommendations** ranked with CLIP visual similarity
- **Your taste profile** built from previous reviews and food photos
- **Lookups** for a named restaurant, vibe, or review

Try: *Find me a moody steakhouse in DTLA* or *Vegetarian recipes that look like the plates I photographed*.
"""
