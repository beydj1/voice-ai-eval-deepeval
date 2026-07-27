from pathlib import Path

from voice_ai_eval.evaluation.engine import EvaluationEngine
from voice_ai_eval.loaders.transcript_loader import load_policy, load_transcript
from voice_ai_eval.models import BatchResult
from voice_ai_eval.providers.mock import MockJudgeProvider
from voice_ai_eval.reporting.report_generator import write_reports


def test_reports_created(tmp_path: Path) -> None:
    result = EvaluationEngine(MockJudgeProvider()).evaluate(load_transcript(Path("data/transcripts/good_appointment.json")), load_policy(Path("data/policies/default.yaml")))
    paths = write_reports(BatchResult(results=[result]), tmp_path)
    assert all(path.exists() for path in paths.values())
