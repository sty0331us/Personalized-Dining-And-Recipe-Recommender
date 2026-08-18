"""Build a structured preference profile from a user's past reviews and photos."""

from __future__ import annotations

from pdr.config import Settings, get_settings
from pdr.data.loaders import catalog
from pdr.data.schemas import UserReview
from pdr.logging_utils import get_logger

logger = get_logger(__name__)


def reviews_for_user(user_id: str | None = None, settings: Settings | None = None) -> list[UserReview]:
    settings = settings or get_settings()
    _, _, reviews, _ = catalog()
    target = user_id or settings.default_user_id
    return [review for review in reviews if review.user_id == target]


def build_preference_profile(
    user_id: str | None = None,
    extra_text: str = "",
    settings: Settings | None = None,
) -> dict:
    """Turn prior visits, ratings, and photo captions into a retrieval query + profile."""
    settings = settings or get_settings()
    reviews = reviews_for_user(user_id, settings)
    liked = [r for r in reviews if (r.rating or 0) >= settings.min_preference_rating]
    disliked = [r for r in reviews if (r.rating or 0) and r.rating < 3.5]
    liked_names = [r.restaurant_name for r in liked if r.restaurant_name]
    captions = [cap for r in liked for cap in r.image_captions]
    review_blob = " ".join(f"{r.title}. {r.text}" for r in liked)
    visual_blob = " ".join(captions)
    retrieval_query = " ".join(
        part for part in (extra_text, review_blob, visual_blob) if part
    ).strip()
    profile = {
        "user_id": user_id or settings.default_user_id,
        "liked_restaurants": liked_names,
        "avoid_or_splurge_only": [r.restaurant_name for r in disliked if r.restaurant_name],
        "review_count": len(reviews),
        "liked_count": len(liked),
        "visual_preferences": captions[:8],
        "preference_query": retrieval_query[:2000],
        "summary": (
            f"User has {len(reviews)} prior California dining reviews. "
            f"High-rated places: {', '.join(liked_names) or 'none yet'}. "
            f"Photo style cues: {'; '.join(captions[:3]) or 'no photos'}."
        ),
    }
    logger.info("Built preference profile for %s (%s reviews)", profile["user_id"], len(reviews))
    return profile
