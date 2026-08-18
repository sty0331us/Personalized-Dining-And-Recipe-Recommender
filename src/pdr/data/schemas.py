"""Pydantic contracts for restaurants, recipes, reviews, and retrieval hits."""

from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field, field_validator


PRICE_SYMBOLS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}


def price_to_symbols(value: Any) -> str:
    if isinstance(value, str) and value.strip().startswith("$"):
        return value.strip()
    try:
        n = int(value)
        return PRICE_SYMBOLS.get(n, "$$")
    except (TypeError, ValueError):
        return "$$"


def parse_image_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (ValueError, SyntaxError):
            if text.startswith("http"):
                return [text]
    return []


class Restaurant(BaseModel):
    item_id: str
    name: str
    location: str = ""
    cuisine: str = ""
    type: str = ""
    rating: float | None = None
    price_range: str = "$$"
    signatures: list[str] = Field(default_factory=list)
    vibes: list[str] = Field(default_factory=list)
    description: str = ""
    environment: str = ""
    shortcomings: list[str] = Field(default_factory=list)

    def to_embedding_text(self) -> str:
        signatures = ", ".join(self.signatures)
        vibes = ", ".join(self.vibes)
        return (
            f"Restaurant: {self.name}\n"
            f"Cuisine: {self.cuisine}\n"
            f"Type: {self.type}\n"
            f"Location: {self.location}\n"
            f"Price: {self.price_range}\n"
            f"Rating: {self.rating}\n"
            f"Signature dishes: {signatures}\n"
            f"Vibes: {vibes}\n"
            f"Setting: {self.environment}\n"
            f"Description: {self.description}"
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.item_id,
            "name": self.name,
            "cuisine": self.cuisine or "",
            "location": self.location or "",
            "price_range": self.price_range or "",
            "rating": float(self.rating or 0.0),
            "source": "restaurant",
            "entity_type": "restaurant",
        }


class Recipe(BaseModel):
    recipe_id: str
    name: str
    cuisine: str = ""
    servings: int | None = None
    prep_time: str = ""
    cook_time: str = ""
    total_time: str = ""
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    image_description: str = ""
    image_path: str = ""

    def to_embedding_text(self) -> str:
        ingredients = ", ".join(self.ingredients[:12])
        return (
            f"Recipe: {self.name}\n"
            f"Cuisine: {self.cuisine}\n"
            f"Time: {self.total_time}\n"
            f"Ingredients: {ingredients}\n"
            f"Visual: {self.image_description}"
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.recipe_id,
            "name": self.name,
            "cuisine": self.cuisine or "",
            "location": "",
            "source": "recipe",
            "entity_type": "recipe",
            "image_path": self.image_path or "",
        }


class UserReview(BaseModel):
    review_id: str
    user_id: str
    item_id: str
    restaurant_name: str = ""
    title: str = ""
    text: str = ""
    rating: float | None = None
    date: str = ""
    image_urls: list[str] = Field(default_factory=list)
    image_captions: list[str] = Field(default_factory=list)

    def to_embedding_text(self) -> str:
        captions = " ".join(self.image_captions)
        return (
            f"Review of {self.restaurant_name}: {self.title}. {self.text} "
            f"Visual notes: {captions}"
        )


class RetrievalHit(BaseModel):
    modality: str
    id: str
    name: str = ""
    cuisine: str = "N/A"
    location: str = "N/A"
    source: str = "N/A"
    entity_type: str = "unknown"
    text_score: float = 0.0
    img_score: float = 0.0
    pref_score: float = 0.0
    fused: float = 0.0
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        return str(value or "")
