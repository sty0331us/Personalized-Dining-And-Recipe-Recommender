"""Load and normalize restaurant, recipe, review, and culinary-map corpora."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pdr.config import Settings, get_settings
from pdr.data.schemas import (
    Recipe,
    Restaurant,
    UserReview,
    parse_image_urls,
    price_to_symbols,
)
from pdr.logging_utils import get_logger

logger = get_logger(__name__)


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Missing data file: %s", path)
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _overlay_by_name(primary: list[dict], overlay: list[dict]) -> list[dict]:
    overlay_index = {str(item.get("name", "")).lower(): item for item in overlay}
    merged = []
    for row in primary:
        extra = overlay_index.get(str(row.get("name", "")).lower(), {})
        merged.append({**row, "_overlay": extra})
    return merged


def _restaurant_from_row(row: dict) -> Restaurant | None:
    name = str(row.get("name", "")).strip()
    if not name:
        return None
    overlay = row.get("_overlay") or {}
    item_id = str(row.get("itemId") or overlay.get("itemId") or f"rest_{name.lower().replace(' ', '_')}")
    signatures = row.get("signatures") or []
    if overlay.get("signature_dish"):
        signatures = list(dict.fromkeys([*signatures, overlay["signature_dish"]]))
    vibes = overlay.get("vibes") or []
    if row.get("vibe"):
        vibes = list(dict.fromkeys([*vibes, row["vibe"]]))
    description = overlay.get("description") or row.get("environment") or ""
    cuisine = overlay.get("cuisine") or row.get("food_style") or row.get("type") or ""
    location = overlay.get("neighborhood") or row.get("location") or ""
    return Restaurant(
        item_id=item_id,
        name=name,
        location=location,
        cuisine=cuisine,
        type=overlay.get("type") or row.get("type") or "",
        rating=row.get("rating") if row.get("rating") is not None else overlay.get("rating"),
        price_range=price_to_symbols(overlay.get("price_range") or row.get("price_range")),
        signatures=[str(s) for s in signatures],
        vibes=[str(v) for v in vibes],
        description=str(description),
        environment=str(row.get("environment") or overlay.get("description") or ""),
        shortcomings=[str(s) for s in (row.get("shortcomings") or [])],
    )


def load_restaurants(settings: Settings | None = None) -> list[Restaurant]:
    settings = settings or get_settings()
    primary = _load_json(settings.restaurants_path)
    overlay = _load_json(settings.restaurant_overlay_path)
    rows = _overlay_by_name(primary, overlay) if overlay else [{**row, "_overlay": {}} for row in primary]
    restaurants = []
    for row in rows:
        parsed = _restaurant_from_row(row)
        if parsed:
            restaurants.append(parsed)
    logger.info("Loaded %s restaurants", len(restaurants))
    return restaurants


def _resolve_recipe_image(recipe_id: str, settings: Settings) -> str:
    candidates = [
        settings.images_dir / f"recipe{recipe_id}.png",
        settings.images_dir / "synthetic_recipe_images" / f"recipe{recipe_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def load_recipes(settings: Settings | None = None) -> list[Recipe]:
    settings = settings or get_settings()
    rows = _load_json(settings.recipes_path)
    recipes = []
    for row in rows:
        recipe_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        if not recipe_id or not name:
            continue
        recipes.append(
            Recipe(
                recipe_id=recipe_id,
                name=name,
                cuisine=str(row.get("cuisine") or ""),
                servings=row.get("servings"),
                prep_time=str(row.get("prep_time") or ""),
                cook_time=str(row.get("cook_time") or ""),
                total_time=str(row.get("total_time") or ""),
                ingredients=[str(x) for x in (row.get("ingredients") or [])],
                directions=[str(x) for x in (row.get("directions") or [])],
                image_description=str(row.get("image_description") or "").strip(" \""),
                image_path=_resolve_recipe_image(recipe_id, settings),
            )
        )
    logger.info("Loaded %s recipes", len(recipes))
    return recipes


def _restaurant_name_index(restaurants: list[Restaurant]) -> dict[str, str]:
    return {r.item_id: r.name for r in restaurants}


def load_reviews(
    settings: Settings | None = None,
    restaurants: list[Restaurant] | None = None,
) -> list[UserReview]:
    settings = settings or get_settings()
    restaurants = restaurants or load_restaurants(settings)
    names = _restaurant_name_index(restaurants)
    narrative = {
        str(item.get("restaurant_name", "")).lower(): item
        for item in _load_json(settings.narrative_reviews_path)
    }
    reviews = []
    for row in _load_json(settings.reviews_path):
        item_id = str(row.get("itemId") or "")
        restaurant_name = names.get(item_id, "")
        extra = narrative.get(restaurant_name.lower(), {})
        captions = list(row.get("image_captions") or [])
        if extra.get("image_description"):
            captions.append(extra["image_description"])
        reviews.append(
            UserReview(
                review_id=str(row.get("reviewId") or f"rev_{item_id}"),
                user_id=str(row.get("userId") or settings.default_user_id),
                item_id=item_id,
                restaurant_name=restaurant_name or extra.get("restaurant_name") or "",
                title=str(row.get("title") or ""),
                text=str(row.get("text") or extra.get("review_text") or ""),
                rating=row.get("rating") if row.get("rating") is not None else extra.get("rating"),
                date=str(row.get("date") or extra.get("visit_date") or ""),
                image_urls=parse_image_urls(row.get("images")),
                image_captions=[str(c) for c in captions if c],
            )
        )
    logger.info("Loaded %s user reviews", len(reviews))
    return reviews


def load_culinary_map(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.culinary_map_path.exists():
        return ""
    return settings.culinary_map_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def catalog() -> tuple[list[Restaurant], list[Recipe], list[UserReview], str]:
    settings = get_settings()
    restaurants = load_restaurants(settings)
    recipes = load_recipes(settings)
    reviews = load_reviews(settings, restaurants)
    culinary_map = load_culinary_map(settings)
    return restaurants, recipes, reviews, culinary_map
