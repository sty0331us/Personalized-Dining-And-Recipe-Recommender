"""Chroma-backed multimodal indexes for restaurants, recipes, dishes, and reviews."""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

from pdr.config import Settings, get_settings
from pdr.data.loaders import catalog
from pdr.logging_utils import get_logger
from pdr.rag.embeddings import EmbeddingService, get_embedding_service

logger = get_logger(__name__)

ARTICLE_COLLECTION = "restaurant_articles"
RECIPE_COLLECTION = "recipes"
IMAGE_COLLECTION = "food_images"
REVIEW_COLLECTION = "user_reviews"


def _chroma_client(settings: Settings):
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _safe_meta(meta: dict) -> dict:
    clean = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def extract_recipe_images(settings: Settings | None = None) -> int:
    """Unzip synthetic recipe photos into data/images if they are not already present."""
    settings = settings or get_settings()
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    existing = list(settings.images_dir.rglob("*.png"))
    if existing:
        return len(existing)
    zip_path = settings.recipe_images_zip
    if not zip_path.exists():
        logger.warning("Recipe image archive not found at %s", zip_path)
        return 0
    logger.info("Extracting recipe images from %s", zip_path)
    with ZipFile(zip_path) as archive:
        archive.extractall(settings.images_dir)
    return len(list(settings.images_dir.rglob("*.png")))


class VectorStore:
    def __init__(self, settings: Settings | None = None, embedder: EmbeddingService | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_service()
        self.client = _chroma_client(self.settings)
        self.articles = self.client.get_or_create_collection(ARTICLE_COLLECTION)
        self.recipes = self.client.get_or_create_collection(RECIPE_COLLECTION)
        self.images = self.client.get_or_create_collection(IMAGE_COLLECTION)
        self.reviews = self.client.get_or_create_collection(REVIEW_COLLECTION)

    def counts(self) -> dict[str, int]:
        return {
            "articles": self.articles.count(),
            "recipes": self.recipes.count(),
            "images": self.images.count(),
            "reviews": self.reviews.count(),
        }

    def rebuild(self, reset: bool = True) -> dict[str, int]:
        extract_recipe_images(self.settings)
        if reset:
            shutil.rmtree(self.settings.chroma_dir, ignore_errors=True)
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.client = _chroma_client(self.settings)
            self.articles = self.client.get_or_create_collection(ARTICLE_COLLECTION)
            self.recipes = self.client.get_or_create_collection(RECIPE_COLLECTION)
            self.images = self.client.get_or_create_collection(IMAGE_COLLECTION)
            self.reviews = self.client.get_or_create_collection(REVIEW_COLLECTION)

        restaurants, recipes, reviews, culinary_map = catalog()

        article_ids, article_docs, article_metas = [], [], []
        for restaurant in restaurants:
            article_ids.append(restaurant.item_id)
            article_docs.append(restaurant.to_embedding_text())
            article_metas.append(_safe_meta(restaurant.as_metadata()))
        if culinary_map:
            for idx, paragraph in enumerate(p.strip() for p in culinary_map.split("\n\n") if p.strip()):
                article_ids.append(f"map_{idx}")
                article_docs.append(paragraph)
                name = (
                    paragraph.split("**")[1].strip()
                    if paragraph.count("**") >= 2
                    else f"map_{idx}"
                )
                article_metas.append(
                    _safe_meta(
                        {
                            "doc_id": f"map_{idx}",
                            "name": name,
                            "cuisine": "",
                            "location": "",
                            "source": "culinary_map",
                            "entity_type": "restaurant",
                        }
                    )
                )
        logger.info("Embedding %s restaurant/map documents", len(article_docs))
        article_vectors = self.embedder.embed_texts(article_docs)
        self.articles.upsert(
            ids=article_ids,
            embeddings=article_vectors.tolist(),
            documents=article_docs,
            metadatas=article_metas,
        )

        recipe_ids, recipe_docs, recipe_metas = [], [], []
        image_ids, image_docs, image_metas, image_paths, image_captions = [], [], [], [], []
        for recipe in recipes:
            recipe_ids.append(f"recipe_{recipe.recipe_id}")
            recipe_docs.append(recipe.to_embedding_text())
            recipe_metas.append(_safe_meta(recipe.as_metadata()))
            caption = recipe.image_description or recipe.name
            image_ids.append(f"dish_{recipe.recipe_id}")
            image_docs.append(caption)
            image_metas.append(
                _safe_meta(
                    {
                        **recipe.as_metadata(),
                        "doc_id": recipe.recipe_id,
                        "source": "recipe_image",
                        "entity_type": "recipe",
                    }
                )
            )
            if recipe.image_path and Path(recipe.image_path).exists():
                image_paths.append(recipe.image_path)
                image_captions.append("")
            else:
                image_paths.append("")
                image_captions.append(caption)

        logger.info("Embedding %s recipes", len(recipe_docs))
        recipe_vectors = self.embedder.embed_texts(recipe_docs)
        self.recipes.upsert(
            ids=recipe_ids,
            embeddings=recipe_vectors.tolist(),
            documents=recipe_docs,
            metadatas=recipe_metas,
        )

        clip_vectors = []
        photo_paths = [p for p in image_paths if p]
        photo_indices = [i for i, p in enumerate(image_paths) if p]
        caption_indices = [i for i, p in enumerate(image_paths) if not p]
        clip_lookup: dict[int, list[float]] = {}
        if photo_paths:
            logger.info("CLIP-embedding %s dish photos", len(photo_paths))
            photo_vecs = self.embedder.embed_images(photo_paths)
            for idx, vec in zip(photo_indices, photo_vecs, strict=True):
                clip_lookup[idx] = vec.tolist()
        captions = [image_captions[i] for i in caption_indices]
        if captions:
            logger.info("CLIP-embedding %s dish captions (text fallback)", len(captions))
            caption_vecs = self.embedder.embed_clip_texts(captions)
            for idx, vec in zip(caption_indices, caption_vecs, strict=True):
                clip_lookup[idx] = vec.tolist()
        for i in range(len(image_ids)):
            clip_vectors.append(clip_lookup[i])
        self.images.upsert(
            ids=image_ids,
            embeddings=clip_vectors,
            documents=image_docs,
            metadatas=image_metas,
        )

        review_ids, review_docs, review_metas = [], [], []
        review_clip_ids, review_clip_docs, review_clip_metas, review_clip_texts = [], [], [], []
        for review in reviews:
            review_ids.append(review.review_id)
            review_docs.append(review.to_embedding_text())
            review_metas.append(
                _safe_meta(
                    {
                        "doc_id": review.item_id,
                        "name": review.restaurant_name,
                        "user_id": review.user_id,
                        "rating": float(review.rating or 0.0),
                        "source": "user_review",
                        "entity_type": "restaurant",
                    }
                )
            )
            for caption_i, caption in enumerate(review.image_captions):
                review_clip_ids.append(f"{review.review_id}_cap_{caption_i}")
                review_clip_docs.append(caption)
                review_clip_texts.append(caption)
                review_clip_metas.append(
                    _safe_meta(
                        {
                            "doc_id": review.item_id,
                            "name": review.restaurant_name,
                            "user_id": review.user_id,
                            "source": "user_photo_caption",
                            "entity_type": "restaurant",
                        }
                    )
                )
        if review_docs:
            logger.info("Embedding %s user reviews", len(review_docs))
            review_vectors = self.embedder.embed_texts(review_docs)
            self.reviews.upsert(
                ids=review_ids,
                embeddings=review_vectors.tolist(),
                documents=review_docs,
                metadatas=review_metas,
            )
        if review_clip_texts:
            logger.info("CLIP-embedding %s user photo captions", len(review_clip_texts))
            review_clip_vecs = self.embedder.embed_clip_texts(review_clip_texts)
            self.images.upsert(
                ids=review_clip_ids,
                embeddings=review_clip_vecs.tolist(),
                documents=review_clip_docs,
                metadatas=review_clip_metas,
            )

        counts = self.counts()
        logger.info("Index rebuild complete: %s", counts)
        return counts


def get_vector_store() -> VectorStore:
    return VectorStore()
