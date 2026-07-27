from pathlib import Path

from voice_ai_eval.evaluation.engine import EvaluationEngine
from voice_ai_eval.loaders.transcript_loader import load_policy, load_transcript
from voice_ai_eval.providers.mock import MockJudgeProvider


def test_engine_pass_and_fail() -> None:
    policy = load_policy(Path("data/policies/default.yaml"))
    engine = EvaluationEngine(MockJudgeProvider(), threshold=0.75)
    good = engine.evaluate(load_transcript(Path("data/transcripts/good_appointment.json")), policy)
    bad = engine.evaluate(load_transcript(Path("data/transcripts/bad_appointment.json")), policy)
    assert good.passed
    assert not bad.passed
    assert bad.critical_failure
