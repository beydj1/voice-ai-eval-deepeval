from voice_ai_eval.providers.base import JudgeProvider
from voice_ai_eval.providers.deepeval_provider import DeepEvalJudgeProvider
from voice_ai_eval.providers.mock import MockJudgeProvider


def create_provider(name: str, model: str, base_url: str | None = None) -> JudgeProvider:
    if name == "mock":
        return MockJudgeProvider()
    if name in {"ollama", "openai"}:
        return DeepEvalJudgeProvider(name, model, base_url)
    raise ValueError(f"Unsupported provider: {name}")
