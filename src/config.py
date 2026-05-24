"""Application-wide typed settings loaded from environment variables.

All configuration lives here; no environment reads happen elsewhere in
``src/``. Settings are validated at import time so a missing required
variable raises immediately rather than surfacing as a late runtime error.

Typical usage::

    from src.config import settings

    print(settings.LOG_LEVEL)
    print(settings.DATABASE_URL)
"""

from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Best-effort: load .env into os.environ so the ai/ providers can read keys via os.getenv().
# On Streamlit Cloud / Docker / production, secrets are already in os.environ, so this is a no-op.
# We guard against missing python-dotenv just in case the package is not installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass


class Settings(BaseSettings):
    """Typed, validated settings sourced from environment variables or ``.env``.

    Field descriptions document each variable's purpose; defaults are chosen
    so the application boots in single-provider mode (SQLite, OpenAI VLM,
    OpenAI embeddings, no failover) without further configuration. Enable
    the secondary VLM by setting ``SECONDARY_LLM_PROVIDER`` to a provider
    name (and supplying its API key). Production deployments override
    values via environment variables or a mounted ``.env`` file.

    Unknown environment variables are silently ignored, allowing the same
    ``.env`` to be shared between services without spurious validation errors.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    IMAGES_DIR: str = Field("data/images", description="Filesystem path for image blobs")
    MAX_IMAGE_BYTES: int = Field(5 * 1024 * 1024, description="Maximum upload size in bytes (5 MB)")

    DATABASE_URL: str = Field(
        "sqlite+aiosqlite:///./dev.db",
        description="DB connection string. SQLite for dev; PostgreSQL for production.",
    )

    LLM_PROVIDER: str = Field("openai", description="openai | gemini | anthropic")
    LLM_MODEL: str = Field("gpt-4o", description="Provider-specific model ID")
    OPENAI_API_KEY: str | None = Field(None)
    GOOGLE_API_KEY: str | None = Field(None)
    ANTHROPIC_API_KEY: str | None = Field(None)

    EMBEDDING_PROVIDER: str = Field("openai", description="openai | gemini")
    EMBEDDING_MODEL: str = Field("text-embedding-3-small")

    SECONDARY_LLM_PROVIDER: str | None = Field(
        None,
        description="Optional fallback VLM provider. Leave unset to disable failover (OpenAI only).",
    )
    SECONDARY_LLM_MODEL: str | None = Field(None)

    SEMAPHORE_LIMIT: int = Field(10, description="Maximum concurrent AI calls")
    AI_CALL_TIMEOUT_S: float = Field(30.0, description="Per-AI-call timeout in seconds")
    AI_CALL_TPM_LIMIT: int = Field(40_000, description="Tokens-per-minute ceiling for the rate limiter")

    LOG_LEVEL: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR")
    ENABLE_TRACING: bool = Field(False, description="Emit OpenTelemetry spans when True")
    OTLP_ENDPOINT: str = Field("http://localhost:4317", description="OTLP exporter endpoint")

    COST_LOG_PATH: str = Field("artefacts/cost.jsonl", description="Append-only cost log path")


settings = Settings()  # type: ignore[call-arg]
"""Process-wide singleton; import this from any module that needs configuration."""

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
