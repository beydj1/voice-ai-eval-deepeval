# Voice AI Eval

A production-oriented evaluation harness for multi-turn voice-agent transcripts. It combines deterministic policy checks with DeepEval's conversational LLM-as-a-judge metrics and generates local JSON, Markdown, and HTML reports.

## What is included

- Multi-turn JSON/YAML transcript ingestion and schema validation
- Versioned YAML policy packs
- Deterministic checks for required tools, required/forbidden phrases, PII patterns, and turn limits
- DeepEval `ConversationalGEval` adapter
- Offline mock mode for free pipeline testing
- Ollama and OpenAI provider modes
- Batch evaluation, CI quality gate, structured reporting, tests, type-friendly package layout

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

## Free offline smoke test

```bash
voice-eval validate data/transcripts --policy data/policies/default.yaml
voice-eval run data/transcripts/good_appointment.json --provider mock
```

Mock mode validates the full pipeline but intentionally does **not** claim to provide real AI judgment.

## Free real local judge with Ollama

Install Ollama, download a capable model, then configure DeepEval's local model integration:

```bash
ollama pull qwen3:8b
deepeval set-ollama --model=qwen3:8b
voice-eval run data/transcripts/good_appointment.json --provider ollama --model qwen3:8b
```

Use a larger model when your computer can support it; judge reliability depends on model capability.

## OpenAI judge

```bash
export OPENAI_API_KEY='...'
voice-eval run data/transcripts --provider openai --model gpt-4.1-mini
```

## Add interview transcripts

Use the structure in `data/transcripts/good_appointment.json`. Required fields are `call_id`, `scenario`, and at least one user and assistant turn. Add the expected outcome whenever it is known. Tool evidence belongs in `tools_called` on the relevant turn.

## Customize business expectations

Copy `data/policies/default.yaml`, version it, and change:

- `llm_criteria`: qualitative whole-conversation rubrics judged by DeepEval
- `deterministic_rules`: objective requirements that should not be delegated to an LLM

Keep critical privacy, authentication, tool invocation, and hard business constraints deterministic whenever evidence is available.

## Commands

```bash
voice-eval validate <file-or-directory> --policy <policy.yaml>
voice-eval run <file-or-directory> --policy <policy.yaml> --provider mock|ollama|openai
voice-eval run data/transcripts --provider mock --threshold 0.75 --fail-below 0.80
```

Exit codes: `0` pass, `1` quality gate failure, `2` configuration/input/runtime error.

## Architecture

```text
File/API adapter -> validated Transcript -> deterministic policy checks
                                  \-> DeepEval conversational metrics
                                     -> aggregate result -> JSON/Markdown/HTML -> CI gate
```

The evaluation engine is intentionally independent of file ingestion. A future FastAPI or webhook adapter can call the same engine without changing judgment logic.

## Important evaluation practice

LLM judges are probabilistic. Production use should include a human-labeled calibration set, judge/version tracking, repeated-run stability checks, disagreement review, cost/latency monitoring, and human review for high-risk failures. Never treat an LLM score as ground truth without calibration.
