from pathlib import Path

from voice_ai_eval.checks.deterministic_checks import run_deterministic_checks
from voice_ai_eval.loaders.transcript_loader import load_policy, load_transcript


def test_good_transcript_passes_rules() -> None:
    results = run_deterministic_checks(load_transcript(Path("data/transcripts/good_appointment.json")), load_policy(Path("data/policies/default.yaml")))
    assert all(item.passed for item in results)


def test_bad_transcript_fails_critical_tool_rule() -> None:
    results = run_deterministic_checks(load_transcript(Path("data/transcripts/bad_appointment.json")), load_policy(Path("data/policies/default.yaml")))
    assert any(not item.passed and item.rule_id == "update-tool-required" for item in results)
