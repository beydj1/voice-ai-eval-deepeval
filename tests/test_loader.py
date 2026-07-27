from pathlib import Path

from voice_ai_eval.loaders.transcript_loader import load_policy, load_transcript


def test_load_examples() -> None:
    transcript = load_transcript(Path("data/transcripts/good_appointment.json"))
    policy = load_policy(Path("data/policies/default.yaml"))
    assert transcript.call_id == "call-good-001"
    assert policy.name == "customer-support-default"
