"""LLM factory supporting IBM watsonx and OpenAI via LangChain."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pdr.config import Settings, get_settings
from pdr.json_utils import parse_json_object
from pdr.logging_utils import get_logger

logger = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    pass


def llm_is_configured(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        return bool(settings.openai_api_key)
    return bool(settings.watsonx_project_id)


def _watsonx_model(settings: Settings) -> BaseChatModel:
    from langchain_ibm import ChatWatsonx

    project_id = settings.watsonx_project_id or None
    if not project_id:
        raise LLMNotConfiguredError(
            "Set WATSONX_PROJECT_ID (and WATSONX_APIKEY) to use the watsonx provider."
        )
    kwargs: dict[str, Any] = {
        "model_id": settings.watsonx_model,
        "url": settings.watsonx_url,
        "project_id": project_id,
        "params": {"temperature": settings.llm_temperature},
    }
    if settings.watsonx_api_key:
        kwargs["apikey"] = settings.watsonx_api_key
    return ChatWatsonx(**kwargs)


def _openai_model(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise LLMNotConfiguredError("Set OPENAI_API_KEY to use the openai provider.")
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        logger.info("Using OpenAI chat model %s", settings.openai_model)
        return _openai_model(settings)
    logger.info("Using watsonx chat model %s", settings.watsonx_model)
    return _watsonx_model(settings)


def invoke_text(system_prompt: str, user_prompt: str, model: BaseChatModel | None = None) -> str:
    chat = model or get_chat_model()
    response = chat.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = response.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ).strip()
    return str(content or "").strip()


def invoke_json(
    system_prompt: str,
    user_prompt: str,
    fallback: Any | None = None,
    model: BaseChatModel | None = None,
) -> Any:
    raw = invoke_text(system_prompt, user_prompt, model=model)
    return parse_json_object(raw, fallback=fallback if fallback is not None else {})
