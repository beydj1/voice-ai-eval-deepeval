from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICE_EVAL_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    provider: Literal["mock", "ollama", "openai"] = "mock"
    model: str = "qwen3:8b"
    threshold: float = Field(default=0.75, ge=0, le=1)
    report_dir: Path = Path("reports")
    fail_on_critical: bool = True
    local_model_base_url: str = "http://localhost:11434"


settings = Settings()
