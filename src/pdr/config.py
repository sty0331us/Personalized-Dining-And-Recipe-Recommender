"""Runtime configuration for the recommender system."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings with safe local defaults."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    images_dir: Path = PROJECT_ROOT / "data" / "images"

    chroma_dir: Path = Field(default=PROJECT_ROOT / "chroma_db", alias="PDR_CHROMA_DIR")
    article_collection: str = "restaurant_articles"
    image_collection: str = "food_images"
    recipe_collection: str = "recipes"
    review_collection: str = "user_reviews"

    text_embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="PDR_TEXT_EMBEDDING_MODEL")
    clip_model: str = Field(default="openai/clip-vit-base-patch32", alias="PDR_CLIP_MODEL")

    fusion_text_weight: float = Field(default=0.55, alias="PDR_FUSION_TEXT_WEIGHT")
    fusion_image_weight: float = Field(default=0.30, alias="PDR_FUSION_IMAGE_WEIGHT")
    fusion_pref_weight: float = Field(default=0.15, alias="PDR_FUSION_PREF_WEIGHT")
    retrieve_k_text: int = 12
    retrieve_k_image: int = 12
    retrieve_k_recipe: int = 12
    rerank_top_n: int = 8

    llm_provider: str = Field(default="watsonx", alias="PDR_LLM_PROVIDER")
    watsonx_url: str = Field(default="https://us-south.ml.cloud.ibm.com", alias="WATSONX_URL")
    watsonx_api_key: str = Field(default="", alias="WATSONX_APIKEY")
    watsonx_project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    watsonx_model: str = Field(default="ibm/granite-4-h-small", alias="PDR_WATSONX_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="PDR_OPENAI_MODEL")
    llm_temperature: float = 0.4
    llm_max_retries: int = 2

    default_user_id: str = Field(default="USER_FUSION_FINDER_99", alias="PDR_DEFAULT_USER_ID")
    min_preference_rating: float = 4.0

    gradio_share: bool = Field(default=False, alias="PDR_GRADIO_SHARE")
    gradio_server_name: str = Field(default="127.0.0.1", alias="PDR_GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(default=7860, alias="PDR_GRADIO_SERVER_PORT")

    restaurants_path: Path = PROJECT_ROOT / "data" / "processed" / "structured_restaurant_data.json"
    restaurant_overlay_path: Path = PROJECT_ROOT / "data" / "processed" / "structured-restaurant-data.json"
    recipes_path: Path = PROJECT_ROOT / "data" / "processed" / "augmented_food_recipe.json"
    reviews_path: Path = PROJECT_ROOT / "data" / "processed" / "augmented_user_review.json"
    narrative_reviews_path: Path = PROJECT_ROOT / "data" / "processed" / "augmented-user-review.json"
    culinary_map_path: Path = PROJECT_ROOT / "data" / "raw" / "California-Culinary-Map.txt"
    recipe_images_zip: Path = PROJECT_ROOT / "data" / "raw" / "synthetic-recipe-images.zip"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
