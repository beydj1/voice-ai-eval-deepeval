from __future__ import annotations

from pathlib import Path

from voice_ai_eval.checks.deterministic_checks import run_deterministic_checks
from voice_ai_eval.models import EvaluationResult, PolicyPack, Severity, Transcript
from voice_ai_eval.providers.base import JudgeProvider


class EvaluationEngine:
    def __init__(self, provider: JudgeProvider, threshold: float = 0.75, fail_on_critical: bool = True) -> None:
        self.provider = provider
        self.threshold = threshold
        self.fail_on_critical = fail_on_critical

    def evaluate(self, transcript: Transcript, policy: PolicyPack, source_file: Path | None = None) -> EvaluationResult:
        deterministic = run_deterministic_checks(transcript, policy)
        metrics = self.provider.evaluate(transcript, policy, self.threshold)

        critical_failure = any(not item.passed and item.severity == Severity.CRITICAL for item in deterministic)
        deterministic_passed = all(item.passed for item in deterministic)
        metric_passed = all(item.passed for item in metrics)
        score = sum(item.score for item in metrics) / len(metrics) if metrics else (1.0 if deterministic_passed else 0.0)
        passed = deterministic_passed and metric_passed and not (self.fail_on_critical and critical_failure)

        return EvaluationResult(
            call_id=transcript.call_id,
            scenario=transcript.scenario,
            policy_name=policy.name,
            policy_version=policy.version,
            overall_score=round(score, 4),
            passed=passed,
            critical_failure=critical_failure,
            deterministic_results=deterministic,
            metric_results=metrics,
            source_file=str(source_file) if source_file else None,
        )
