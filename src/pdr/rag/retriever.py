"""Similarity retrieval over text, CLIP image, recipe, and review collections."""

from __future__ import annotations

import numpy as np

from pdr.config import Settings, get_settings
from pdr.data.schemas import RetrievalHit
from pdr.rag.embeddings import EmbeddingService, get_embedding_service
from pdr.rag.fusion import distance_to_similarity, fuse_hits
from pdr.rag.indexer import VectorStore, get_vector_store


def _unwrap(result: dict) -> tuple[list, list, list, list]:
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return ids, docs, metas, dists


def _hits_from_query(
    result: dict,
    modality: str,
    score_field: str,
) -> list[RetrievalHit]:
    ids, docs, metas, dists = _unwrap(result)
    sims = distance_to_similarity(dists) if dists else np.zeros(len(ids), dtype=np.float32)
    hits = []
    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        score = float(sims[i]) if i < len(sims) else 0.0
        payload = {
            "modality": modality,
            "id": str(meta.get("doc_id") or doc_id),
            "name": str(meta.get("name") or ""),
            "cuisine": str(meta.get("cuisine") or "N/A"),
            "location": str(meta.get("location") or "N/A"),
            "source": str(meta.get("source") or "N/A"),
            "entity_type": str(meta.get("entity_type") or "unknown"),
            "snippet": (docs[i] or "").replace("\n", " ").strip() if i < len(docs) else "",
            "metadata": meta,
        }
        payload[score_field] = score
        hits.append(RetrievalHit(**payload))
    return hits


class MultimodalRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        self.embedder = embedder or get_embedding_service()

    def retrieve_articles(self, query: str, k: int | None = None, where: dict | None = None) -> list[RetrievalHit]:
        vector = self.embedder.embed_texts([query])[0]
        result = self.store.articles.query(
            query_embeddings=[vector.tolist()],
            n_results=k or self.settings.retrieve_k_text,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, modality="article", score_field="text_score")

    def retrieve_recipes(self, query: str, k: int | None = None, where: dict | None = None) -> list[RetrievalHit]:
        vector = self.embedder.embed_texts([query])[0]
        result = self.store.recipes.query(
            query_embeddings=[vector.tolist()],
            n_results=k or self.settings.retrieve_k_recipe,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, modality="recipe", score_field="text_score")

    def retrieve_images_by_text(self, query: str, k: int | None = None, where: dict | None = None) -> list[RetrievalHit]:
        vector = self.embedder.embed_clip_texts([query])[0]
        result = self.store.images.query(
            query_embeddings=[vector.tolist()],
            n_results=k or self.settings.retrieve_k_image,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, modality="image", score_field="img_score")

    def retrieve_images_by_image(self, image, k: int | None = None) -> list[RetrievalHit]:
        vector = self.embedder.embed_pil_image(image)
        result = self.store.images.query(
            query_embeddings=[vector.tolist()],
            n_results=k or self.settings.retrieve_k_image,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, modality="image", score_field="img_score")

    def retrieve_reviews(self, query: str, k: int = 8, where: dict | None = None) -> list[RetrievalHit]:
        vector = self.embedder.embed_texts([query])[0]
        result = self.store.reviews.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, modality="review", score_field="pref_score")

    def search(
        self,
        query: str,
        where_text: dict | None = None,
        where_img: dict | None = None,
        preference_query: str | None = None,
        image=None,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        text_hits = self.retrieve_articles(query, where=where_text)
        image_hits = self.retrieve_images_by_text(query, where=where_img)
        if image is not None:
            image_hits = image_hits + self.retrieve_images_by_image(image)
        pref_hits = self.retrieve_reviews(preference_query or query) if preference_query or query else []
        return fuse_hits(
            text_hits=text_hits,
            image_hits=image_hits,
            preference_hits=pref_hits,
            w_text=self.settings.fusion_text_weight,
            w_img=self.settings.fusion_image_weight,
            w_pref=self.settings.fusion_pref_weight,
            top_n=top_n or self.settings.rerank_top_n,
        )

    def search_recipes(self, query: str, preference_query: str | None = None) -> list[RetrievalHit]:
        recipe_hits = self.retrieve_recipes(query)
        image_hits = self.retrieve_images_by_text(query)
        pref_hits = self.retrieve_reviews(preference_query or query) if preference_query else []
        return fuse_hits(
            text_hits=recipe_hits,
            image_hits=image_hits,
            preference_hits=pref_hits,
            w_text=self.settings.fusion_text_weight,
            w_img=self.settings.fusion_image_weight,
            w_pref=self.settings.fusion_pref_weight,
            top_n=self.settings.rerank_top_n,
        )
