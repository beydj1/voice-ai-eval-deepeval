"""Framework-wide constants for Voice AI evaluation."""

DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODEL = "gpt-4.1"

SUPPORTED_JUDGE_PROVIDERS = {
    "mock",
    "ollama",
    "openai",
}