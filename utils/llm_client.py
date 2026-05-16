"""Unified chat-completion client for the PoT replication scripts."""

from __future__ import annotations

import random
import time
from typing import Optional

import config


def get_completion(
    prompt: str,
    system_prompt: str = "",
    model: Optional[str] = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    n: int = 1,
) -> list[str]:
    """Get n completions from the configured provider."""
    provider = config.DEFAULT_PROVIDER
    config.validate_provider(provider)

    model = model or config.DEFAULT_MODEL
    max_tokens = max_tokens or config.MAX_TOKENS
    temperature = temperature if temperature is not None else config.TEMPERATURE

    for attempt in range(config.MAX_RETRIES):
        try:
            if provider == "gemini":
                return _gemini_call(prompt, system_prompt, model, max_tokens, temperature, n)
            if provider == "groq":
                return _groq_call(prompt, system_prompt, model, max_tokens, temperature, n)
            if provider == "openrouter":
                return _openrouter_call(prompt, system_prompt, model, max_tokens, temperature, n)
            if provider == "anthropic":
                return _anthropic_call(prompt, system_prompt, model, max_tokens, temperature, n)
            if provider == "openai":
                return _openai_call(prompt, system_prompt, model, max_tokens, temperature, n)
            raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:  # noqa: BLE001
            if _is_non_retryable_api_error(exc):
                raise RuntimeError(
                    f"{provider} API call failed with model {model!r}: {exc}"
                ) from exc
            if attempt >= config.MAX_RETRIES - 1:
                raise RuntimeError(
                    f"{provider} API call failed after {config.MAX_RETRIES} attempts "
                    f"with model {model!r}: {exc}"
                ) from exc
            wait = (2**attempt) + random.uniform(0, 1)
            print(f"API error from {provider} (attempt {attempt + 1}): {exc}")
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return [""] * n


def _is_non_retryable_api_error(exc: Exception) -> bool:
    text = str(exc).lower()
    permanent_markers = [
        "insufficient_quota",
        "invalid api key",
        "incorrect api key",
        "authentication",
        "unauthorized",
        "permission",
        "model_not_found",
        "does not exist",
    ]
    return any(marker in text for marker in permanent_markers)


def _messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _gemini_call(prompt, system_prompt, model, max_tokens, temperature, n) -> list[str]:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        candidate_count=1,
    )
    gemini_model = genai.GenerativeModel(model)

    results = []
    for _ in range(n):
        response = gemini_model.generate_content(full_prompt, generation_config=generation_config)
        results.append(getattr(response, "text", "") or "")
        if n > 1:
            time.sleep(0.5)
    return results


def _groq_call(prompt, system_prompt, model, max_tokens, temperature, n) -> list[str]:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    results = []
    for _ in range(n):
        response = client.chat.completions.create(
            model=model,
            messages=_messages(prompt, system_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(response.choices[0].message.content or "")
        if n > 1:
            time.sleep(0.3)
    return results


def _openrouter_call(prompt, system_prompt, model, max_tokens, temperature, n) -> list[str]:
    import openai

    client = openai.OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=_messages(prompt, system_prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        extra_headers={
            "HTTP-Referer": "https://github.com/TIGER-AI-Lab/Program-of-Thoughts",
            "X-Title": "Program of Thoughts Replication",
        },
    )
    return [choice.message.content or "" for choice in response.choices]


def _anthropic_call(prompt, system_prompt, model, max_tokens, temperature, n) -> list[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    results = []
    for _ in range(n):
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        results.append(response.content[0].text)
        if n > 1:
            time.sleep(0.3)
    return results


def _openai_call(prompt, system_prompt, model, max_tokens, temperature, n) -> list[str]:
    import openai

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=_messages(prompt, system_prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    return [choice.message.content or "" for choice in response.choices]
