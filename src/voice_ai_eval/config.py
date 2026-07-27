from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_ai_eval.constants import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
)


class Settings(BaseSettings):
    """
    Runtime configuration for the Voice AI evaluation framework.

    Values may be overridden through:

    - .env
    - .env.local
    - environment variables
    - CLI arguments

    Environment variables use the VOICE_EVAL_ prefix.

    Examples:

        VOICE_EVAL_PROVIDER=openai
        VOICE_EVAL_MODEL=gpt-4.1
    """

    model_config = SettingsConfigDict(
        env_prefix="VOICE_EVAL_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    provider: Literal["mock", "ollama", "openai"] = DEFAULT_JUDGE_PROVIDER
    model: str = DEFAULT_JUDGE_MODEL

    threshold: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )

    report_dir: Path = Path("reports")
    fail_on_critical: bool = True

    local_model_base_url: str = "http://localhost:11434"


settings = Settings()