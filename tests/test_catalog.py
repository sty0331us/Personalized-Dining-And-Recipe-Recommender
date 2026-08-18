from pdr.data.loaders import load_recipes, load_restaurants, load_reviews
from pdr.preference.engine import build_preference_profile


def test_restaurants_load_and_merge_overlay():
    restaurants = load_restaurants()
    assert len(restaurants) >= 200
    iron = next(r for r in restaurants if r.name == "Iron & Embers")
    assert iron.location
    assert "moody" in " ".join(iron.vibes).lower() or "moody" in iron.environment.lower()
    assert iron.price_range in {"$", "$$", "$$$", "$$$$"}


def test_recipes_include_visual_descriptions():
    recipes = load_recipes()
    assert len(recipes) >= 100
    first = recipes[0]
    assert first.name
    assert first.image_description


def test_reviews_attach_restaurant_names_and_captions():
    reviews = load_reviews()
    assert reviews
    sample = reviews[0]
    assert sample.user_id
    assert sample.restaurant_name
    assert sample.text


def test_preference_profile_uses_default_user():
    profile = build_preference_profile()
    assert profile["review_count"] >= 1
    assert profile["liked_restaurants"]
    assert profile["preference_query"]
