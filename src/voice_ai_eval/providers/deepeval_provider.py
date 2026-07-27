from __future__ import annotations

import os
from typing import Any

from voice_ai_eval.models import MetricResult, PolicyPack, Transcript
from voice_ai_eval.providers.base import JudgeProvider


class DeepEvalJudgeProvider(JudgeProvider):
    """DeepEval adapter supporting OpenAI defaults or locally configured Ollama."""

    def __init__(self, provider: str, model: str, base_url: str | None = None) -> None:
        self.name = provider
        self.model_name = model
        self.base_url = base_url

    def _model(self) -> str | Any:
        if self.name == "ollama":
            if self.base_url:
                os.environ.setdefault("LOCAL_MODEL_BASE_URL", self.base_url)
            # DeepEval's Ollama integration is configured through `deepeval set-ollama`.
            # Returning the model name lets DeepEval use the selected local model configuration.
            return self.model_name
        return self.model_name

    def evaluate(self, transcript: Transcript, policy: PolicyPack, threshold: float) -> list[MetricResult]:
        try:
            from deepeval.metrics import ConversationalGEval
            from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn
        except ImportError as exc:
            raise RuntimeError("DeepEval is not installed. Run: pip install -e '.[dev]'") from exc

        turns = [
            Turn(
                role=turn.role.value,
                content=turn.content,
                tools_called=turn.tools_called or None,
            )
            for turn in transcript.turns
        ]
        case = ConversationalTestCase(
            turns=turns,
            scenario=transcript.scenario,
            expected_outcome=transcript.expected_outcome,
        )

        results: list[MetricResult] = []
        for name, criteria in policy.llm_criteria.items():
            contextual_criteria = (
                f"{criteria}\n\nScenario: {transcript.scenario}\n"
                f"Expected outcome: {transcript.expected_outcome or 'Not supplied'}"
            )
            metric = ConversationalGEval(
                name=name,
                criteria=contextual_criteria,
                evaluation_params=[MultiTurnParams.CONTENT],
                threshold=threshold,
                model=self._model(),
                async_mode=False,
                verbose_mode=False,
            )
            metric.measure(case)
            score = float(metric.score or 0.0)
            results.append(MetricResult(name=name, score=score, passed=bool(metric.is_successful()), reason=str(metric.reason or "No reason returned"), provider=self.name))
        return results
