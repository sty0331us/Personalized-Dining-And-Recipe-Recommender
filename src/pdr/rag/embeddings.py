"""Sentence-Transformers (text) and CLIP (image / cross-modal) embedders."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pdr.config import Settings, get_settings
from pdr.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Lazy-loaded dual encoder: MiniLM 384-d text + CLIP 512-d vision/text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._text_model = None
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def _load_text(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading text embedder %s", self.settings.text_embedding_model)
            self._text_model = SentenceTransformer(self.settings.text_embedding_model)
        return self._text_model

    def _load_clip(self):
        if self._clip_model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading CLIP %s on %s", self.settings.clip_model, self._device)
            self._clip_model = CLIPModel.from_pretrained(self.settings.clip_model).to(self._device)
            self._clip_processor = CLIPProcessor.from_pretrained(self.settings.clip_model, use_fast=True)
            self._clip_model.eval()
        return self._clip_model, self._clip_processor

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load_text()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_clip_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 512), dtype=np.float32)
        model, processor = self._load_clip()
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self._device)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def embed_images(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        import torch
        from PIL import Image

        if not paths:
            return np.zeros((0, 512), dtype=np.float32)
        model, processor = self._load_clip()
        chunks = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            images = []
            for path in batch:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(self._device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def embed_pil_image(self, image) -> np.ndarray:
        import torch

        model, processor = self._load_clip()
        rgb = image.convert("RGB")
        inputs = processor(images=[rgb], return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def recipe_image_path(recipe_id: str, settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    for candidate in (
        settings.images_dir / f"recipe{recipe_id}.png",
        settings.images_dir / "synthetic_recipe_images" / f"recipe{recipe_id}.png",
    ):
        if candidate.exists():
            return candidate
    return None
