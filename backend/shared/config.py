"""Unified configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))

    @property
    def dimension(self) -> int:
        dimensions: dict[str, int] = {
            "BAAI/bge-m3": 1024,
        }
        return dimensions.get(self.model, 1024)


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://ema:ema123@localhost:5432/memory")
    )
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))


config = AppConfig()
