from voice_ai_eval.constants import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    SUPPORTED_JUDGE_PROVIDERS,
)
from voice_ai_eval.providers.base import JudgeProvider
from voice_ai_eval.providers.deepeval_provider import DeepEvalJudgeProvider
from voice_ai_eval.providers.mock import MockJudgeProvider
from voice_ai_eval.providers.openai_structured import (
    OpenAIStructuredJudgeProvider,
)


def create_provider(
    name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> JudgeProvider:
    """
    Create the configured LLM judge provider.

    When no provider or model is explicitly supplied, the framework uses:

        provider: openai
        model: gpt-4.1

    CLI arguments and environment configuration may still override these
    defaults.

    Supported provider behavior:

        mock:
            Returns deterministic fake evaluation results without making
            external API requests.

        openai:
            Uses DeepEval with an OpenAI judge model.

        ollama:
            Uses DeepEval with an Ollama-hosted judge model.

        openai-structured:
            Uses one OpenAI structured-output request per transcript to
            evaluate all supported LLM criteria together.
    """
    resolved_name = name or DEFAULT_JUDGE_PROVIDER
    resolved_model = model or DEFAULT_JUDGE_MODEL

    if resolved_name not in SUPPORTED_JUDGE_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_JUDGE_PROVIDERS))
        raise ValueError(
            f"Unsupported provider: {resolved_name}. "
            f"Supported providers: {supported}"
        )

    if resolved_name == "mock":
        return MockJudgeProvider()

    if resolved_name == "openai-structured":
        return OpenAIStructuredJudgeProvider(
            model=resolved_model,
            base_url=base_url,
        )

    if resolved_name in {"ollama", "openai"}:
        return DeepEvalJudgeProvider(
            provider=resolved_name,
            model=resolved_model,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported provider: {resolved_name}")

