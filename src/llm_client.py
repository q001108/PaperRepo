from __future__ import annotations

import os
from typing import Any

import requests


SUPPORTED_LLM_PROVIDERS = {"openai", "deepseek"}


def strip_trailing_slashes(value: str) -> str:
    return value.strip().rstrip("/")


def chat_completions_url(provider: str) -> str:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "deepseek":
        base_url = strip_trailing_slashes(os.getenv("DEEPSEEK_BASE_URL", "").strip())
        if not base_url:
            base_url = "https://api.deepseek.com"
        return f"{base_url}/chat/completions"

    base_url = strip_trailing_slashes(os.getenv("OPENAI_BASE_URL", "").strip())
    if not base_url:
        base_url = "https://api.openai.com/v1"
    return f"{base_url}/chat/completions"


def llm_api_key(provider: str) -> str:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_llm_configured(provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower()
    return bool(
        normalized_provider in SUPPORTED_LLM_PROVIDERS
        and model.strip()
        and llm_api_key(normalized_provider)
    )


def chat_completion_content(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    response_format_json: bool = False,
    max_tokens: int = 700,
    temperature: float = 0,
    timeout: int = 30,
) -> str:
    normalized_provider = provider.strip().lower()
    if not is_llm_configured(normalized_provider, model):
        raise ValueError("LLM provider, model, or API key is not configured.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        chat_completions_url(normalized_provider),
        headers={
            "Authorization": f"Bearer {llm_api_key(normalized_provider)}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])
