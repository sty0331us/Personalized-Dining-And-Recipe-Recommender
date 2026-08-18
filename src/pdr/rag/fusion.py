"""Normalize scores and fuse text, image, and preference rankings."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pdr.data.schemas import RetrievalHit


def minmax(values: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], treating a constant array as equal confidence."""
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def distance_to_similarity(distances: list[float] | np.ndarray) -> np.ndarray:
    """Convert Chroma cosine distance to similarity."""
    return 1.0 - np.asarray(distances, dtype=np.float32)


def fuse_hits(
    text_hits: list[RetrievalHit],
    image_hits: list[RetrievalHit],
    preference_hits: list[RetrievalHit] | None = None,
    w_text: float = 0.55,
    w_img: float = 0.30,
    w_pref: float = 0.15,
    top_n: int = 8,
) -> list[RetrievalHit]:
    """Entity-aware weighted fusion with min-max calibration per modality.

    Hits that share an entity id are merged so a restaurant that matches both
    California copy and a dish photo ranks above a single-modality match.
    """
    buckets: dict[str, RetrievalHit] = {}

    def _ingest(hits: list[RetrievalHit], field: str, scores: np.ndarray) -> None:
        for hit, score in zip(hits, scores, strict=False):
            name_key = (hit.name or "").strip().lower()
            key = name_key or hit.id or f"{hit.modality}:{hit.snippet[:24]}"
            current = buckets.get(key)
            if current is None:
                current = hit.model_copy(deep=True)
                buckets[key] = current
            setattr(current, field, max(float(getattr(current, field)), float(score)))
            if hit.snippet and len(hit.snippet) > len(current.snippet):
                current.snippet = hit.snippet
            current.metadata.update(hit.metadata)

    t_norm = minmax([h.text_score for h in text_hits]) if text_hits else np.array([])
    i_norm = minmax([h.img_score for h in image_hits]) if image_hits else np.array([])
    p_hits = preference_hits or []
    p_norm = minmax([h.pref_score for h in p_hits]) if p_hits else np.array([])

    _ingest(text_hits, "text_score", t_norm)
    _ingest(image_hits, "img_score", i_norm)
    _ingest(p_hits, "pref_score", p_norm)

    fused = []
    for hit in buckets.values():
        hit.fused = float(
            w_text * hit.text_score + w_img * hit.img_score + w_pref * hit.pref_score
        )
        fused.append(hit)
    fused.sort(key=lambda row: row.fused, reverse=True)
    return fused[: max(0, min(int(top_n), len(fused)))]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    k: int = 60,
    top_n: int = 8,
) -> list[RetrievalHit]:
    """Optional RRF reranker used when fusion weights should stay rank-based."""
    scores: dict[str, float] = defaultdict(float)
    canonical: dict[str, RetrievalHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.id] += 1.0 / (k + rank)
            canonical.setdefault(hit.id, hit)
    ordered = sorted(canonical.values(), key=lambda h: scores[h.id], reverse=True)
    for hit in ordered:
        hit.fused = float(scores[hit.id])
    return ordered[:top_n]
