"""Project configuration for the Program-of-Thoughts replication.

Secrets are read from environment variables or a local .env file. Keep real API
keys out of this file so the project is safe to share and rerun.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if value in {"...", "sk-your-openai-key", "sk-ant-your-anthropic-key"}:
        return ""
    return value


_load_dotenv()

GEMINI_API_KEY = _env("GEMINI_API_KEY")
GROQ_API_KEY = _env("GROQ_API_KEY")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _env("OPENAI_API_KEY")

_CONFIGURED_KEYS = {
    "gemini": GEMINI_API_KEY,
    "groq": GROQ_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
}


def _default_provider() -> str:
    explicit = _env("POT_PROVIDER")
    if explicit:
        return explicit.lower()
    for provider in ("gemini", "groq", "openrouter", "openai", "anthropic"):
        if _CONFIGURED_KEYS[provider]:
            return provider
    return "groq"


DEFAULT_PROVIDER = _default_provider()

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.1-8b-instant",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
    "anthropic": "claude-3-5-haiku-latest",
    "openai": "gpt-4o-mini",
}
_MODEL_HINT_PREFIXES = {
    "gemini": ("gemini-",),
    "groq": ("llama-", "openai/", "qwen/", "deepseek-", "moonshotai/"),
    "openrouter": ("meta-llama/", "google/", "openai/", "anthropic/", "qwen/", "deepseek/"),
    "anthropic": ("claude-",),
    "openai": ("gpt-", "o1", "o3", "o4"),
}


def _default_model() -> str:
    env_model = _env("POT_MODEL")
    fallback = DEFAULT_MODELS.get(DEFAULT_PROVIDER, "llama-3.1-8b-instant")
    if not env_model:
        return fallback
    prefixes = _MODEL_HINT_PREFIXES.get(DEFAULT_PROVIDER, ())
    if prefixes and not env_model.startswith(prefixes):
        return fallback
    return env_model


DEFAULT_MODEL = _default_model()

MAX_TOKENS = int(_env("POT_MAX_TOKENS", "1024"))
TEMPERATURE = float(_env("POT_TEMPERATURE", "0.0"))
SC_SAMPLES = int(_env("POT_SC_SAMPLES", "10"))
CODE_TIMEOUT_SECS = int(_env("POT_CODE_TIMEOUT_SECS", "10"))
MAX_RETRIES = int(_env("POT_MAX_RETRIES", "3"))

OUTPUT_DIR = _env("POT_OUTPUT_DIR", "outputs")
DATA_DIR = _env("POT_DATA_DIR", "data")


PROVIDER_KEYS = _CONFIGURED_KEYS


def validate_provider(provider: str | None = None) -> None:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDER_KEYS:
        choices = ", ".join(PROVIDER_KEYS)
        raise ValueError(f"Unknown POT_PROVIDER={provider!r}. Choose one of: {choices}.")
    if not PROVIDER_KEYS[provider]:
        env_name = f"{provider.upper()}_API_KEY"
        raise RuntimeError(
            f"Missing {env_name}. Add it to .env or your shell environment, "
            f"or set POT_PROVIDER to a provider you have configured."
        )
