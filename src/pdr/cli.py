"""Command-line entry points."""

from __future__ import annotations

from pdr.rag.indexer import VectorStore, extract_recipe_images


def ingest_main() -> None:
    extract_recipe_images()
    counts = VectorStore().rebuild(reset=True)
    print("Indexed collections:", counts)
