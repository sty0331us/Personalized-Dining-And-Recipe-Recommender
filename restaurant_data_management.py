"""Interactive catalog manager, now backed by the normalized restaurant schema."""

from __future__ import annotations

from pdr.config import get_settings
from pdr.data.loaders import load_restaurants


def manage_restaurants() -> None:
    settings = get_settings()
    restaurants = load_restaurants(settings)
    print(f"\nRESTAURANT DATABASE | Records: {len(restaurants)}")
    for i, restaurant in enumerate(restaurants[:25]):
        print(f"{i}: {restaurant.name} — {restaurant.location} ({restaurant.price_range})")
    print("... use the Gradio Add Restaurant tab for writes, then re-run ingest.")
    print(f"Source file: {settings.restaurants_path}")


if __name__ == "__main__":
    manage_restaurants()
