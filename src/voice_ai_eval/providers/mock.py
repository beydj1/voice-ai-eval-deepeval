from voice_ai_eval.models import MetricResult, PolicyPack, Transcript
from voice_ai_eval.providers.base import JudgeProvider


class MockJudgeProvider(JudgeProvider):
    """Offline provider for exercising the complete pipeline without claiming real AI judgment."""

    name = "mock"

    def evaluate(self, transcript: Transcript, policy: PolicyPack, threshold: float) -> list[MetricResult]:
        results: list[MetricResult] = []
        for name in policy.llm_criteria:
            normalized = name.casefold().replace(" ", "_")
            score = float(transcript.metadata.get("mock_scores", {}).get(normalized, 0.80))
            results.append(MetricResult(name=name, score=score, passed=score >= threshold, reason="Simulated score from offline mock provider; not a real LLM judgment.", provider=self.name))
        return results
