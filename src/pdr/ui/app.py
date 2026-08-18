"""Gradio interface for the personalized multimodal dining recommender."""

from __future__ import annotations

import json
import gradio as gr

from pdr.config import get_settings
from pdr.data.loaders import catalog
from pdr.preference.engine import build_preference_profile
from pdr.ui.orchestrator import handle_user_turn

CUSTOM_CSS = """
.gradio-container {max-width: 1100px !important;}
"""


def _chat(message: str, history: list, user_id: str, image) -> tuple[list, str]:
    if not message or not str(message).strip():
        return history, ""
    reply = handle_user_turn(message.strip(), user_id=user_id or None, uploaded_image=image)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def _preference_markdown(user_id: str) -> str:
    profile = build_preference_profile(user_id or None)
    liked = ", ".join(profile.get("liked_restaurants") or []) or "—"
    visuals = profile.get("visual_preferences") or []
    visual_block = "\n".join(f"- {cap}" for cap in visuals[:5]) or "- No photo captions yet"
    return (
        f"**User:** `{profile.get('user_id')}`  \n"
        f"**Reviews:** {profile.get('review_count')} · **Liked:** {profile.get('liked_count')}  \n"
        f"**High-rated restaurants:** {liked}\n\n"
        f"**Photo / plating cues**\n{visual_block}"
    )


def _add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    if not name.strip():
        return "Restaurant name is required."
    settings = get_settings()
    path = settings.restaurant_overlay_path
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows.append(
        {
            "name": name.strip(),
            "neighborhood": location.strip(),
            "cuisine": cuisine.strip(),
            "type": "user-added",
            "rating": None,
            "price_range": price,
            "signature_dish": "",
            "vibes": [],
            "description": description.strip(),
        }
    )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    catalog.cache_clear()
    return f"Added **{name}**. Re-run `python scripts/ingest.py` to refresh the vector index."


def _add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    if not name.strip():
        return "Recipe name is required."
    settings = get_settings()
    path = settings.recipes_path
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    next_id = max((int(r.get("id") or 0) for r in rows), default=0) + 1
    rows.append(
        {
            "id": next_id,
            "name": name.strip(),
            "cuisine": cuisine.strip(),
            "servings": 2,
            "prep_time": prep_time.strip() or "20 mins",
            "cook_time": "20 mins",
            "total_time": prep_time.strip() or difficulty,
            "ingredients": [line.strip() for line in ingredients.split("\n") if line.strip()],
            "directions": [line.strip() for line in instructions.split("\n") if line.strip()],
            "image_description": name.strip(),
        }
    )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    catalog.cache_clear()
    return f"Added **{name}**. Re-run `python scripts/ingest.py` to refresh the vector index."


def build_demo() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(
        title="Connoisseur Companion",
        theme=gr.themes.Soft(primary_hue="stone", secondary_hue="orange"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown(
            """
# Connoisseur Companion
Personalized California dining and recipe recommendations with **multimodal RAG**, **review/photo preference**, and **LangGraph multi-agent orchestration**.
"""
        )
        user_id = gr.Textbox(
            value=settings.default_user_id,
            label="User ID",
            info="Reviews and dish photos for this user personalize retrieval and reranking.",
        )
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=520, type="messages", label="Connoisseur Companion")
                image = gr.Image(type="pil", label="Optional dish photo (CLIP visual query)")
                msg = gr.Textbox(
                    label="Ask for restaurants, recipes, or a vibe",
                    placeholder='e.g. "Find me a zen omakase in Little Tokyo" or "Weeknight recipes like the plates I photographed"',
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                examples = gr.Examples(
                    examples=[
                        ["Find me a moody restaurant in DTLA"],
                        ["Tell me about Iron & Embers"],
                        ["Vegetarian recipes I can cook this week"],
                        ["Romantic wine bar energy in Pasadena"],
                    ],
                    inputs=[msg],
                )
                send.click(_chat, [msg, chatbot, user_id, image], [chatbot, msg])
                msg.submit(_chat, [msg, chatbot, user_id, image], [chatbot, msg])
                clear.click(lambda: ([], ""), None, [chatbot, msg])

            with gr.Tab("Your taste profile"):
                profile_md = gr.Markdown(_preference_markdown(settings.default_user_id))
                refresh = gr.Button("Refresh profile")
                refresh.click(_preference_markdown, [user_id], [profile_md])
                user_id.change(_preference_markdown, [user_id], [profile_md])

            with gr.Tab("Add restaurant"):
                with gr.Row():
                    r_name = gr.Textbox(label="Name")
                    r_cuisine = gr.Textbox(label="Cuisine")
                    r_price = gr.Dropdown(["$", "$$", "$$$", "$$$$"], value="$$", label="Price")
                r_location = gr.Textbox(label="Neighborhood / city")
                r_desc = gr.Textbox(label="Description", lines=3)
                r_btn = gr.Button("Add restaurant", variant="primary")
                r_out = gr.Markdown()
                r_btn.click(_add_restaurant, [r_name, r_cuisine, r_price, r_location, r_desc], [r_out])

            with gr.Tab("Add recipe"):
                with gr.Row():
                    p_name = gr.Textbox(label="Name")
                    p_cuisine = gr.Textbox(label="Cuisine")
                    p_diff = gr.Dropdown(["Easy", "Medium", "Hard"], value="Easy", label="Difficulty")
                p_time = gr.Textbox(label="Prep time", value="30 mins")
                p_ing = gr.Textbox(label="Ingredients (one per line)", lines=4)
                p_dir = gr.Textbox(label="Directions (one per line)", lines=4)
                p_btn = gr.Button("Add recipe", variant="primary")
                p_out = gr.Markdown()
                p_btn.click(_add_recipe, [p_name, p_cuisine, p_diff, p_time, p_ing, p_dir], [p_out])

            with gr.Tab("How it works"):
                gr.Markdown(
                    """
## Retrieval
1. MiniLM embeds California restaurant copy, recipes, and your reviews.
2. CLIP embeds dish photos (or photo captions when a file is missing).
3. Scores are min-max calibrated per modality, then fused:

`s_fused = w_text · s_text + w_img · s_img + w_pref · s_pref`

## Agents
LangGraph runs profile generation and multimodal retrieval sequentially, then fans out Trend / Style / Nutrition in parallel before the Recommendation Expert synthesizes the answer.

MCP tools (`get_restaurant_info`, `recommend_by_vibe`, `get_review`, `multimodal_search`, `personalized_recommend`) are available to this UI and to external hosts.
"""
                )
                _ = examples
        return demo


def main() -> None:
    settings = get_settings()
    demo = build_demo()
    demo.launch(
        share=settings.gradio_share,
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
