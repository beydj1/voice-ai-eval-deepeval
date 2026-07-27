.PHONY: install test lint validate demo clean
install:
	python -m pip install -e '.[dev]'
test:
	pytest --cov=voice_ai_eval --cov-report=term-missing
lint:
	ruff check .
validate:
	voice-eval validate data/transcripts --policy data/policies/default.yaml
demo:
	voice-eval run data/transcripts/good_appointment.json --provider mock
clean:
	rm -rf reports/* .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	touch reports/.gitkeep
