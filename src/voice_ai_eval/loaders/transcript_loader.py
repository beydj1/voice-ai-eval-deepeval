from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from voice_ai_eval.models import PolicyPack, Transcript


class LoadError(ValueError):
    """Raised when an input file cannot be loaded or validated."""


def _read_structured(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            value = json.loads(raw)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(raw)
        else:
            raise LoadError(f"Unsupported file type: {path.suffix}")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise LoadError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LoadError(f"Expected an object at the root of {path}")
    return value


def load_transcript(path: Path) -> Transcript:
    try:
        return Transcript.model_validate(_read_structured(path))
    except Exception as exc:
        if isinstance(exc, LoadError):
            raise
        raise LoadError(f"Invalid transcript {path}: {exc}") from exc


def load_transcripts(path: Path) -> list[tuple[Path, Transcript]]:
    if path.is_file():
        return [(path, load_transcript(path))]
    if not path.is_dir():
        raise LoadError(f"Input does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".yaml", ".yml"})
    if not files:
        raise LoadError(f"No JSON/YAML transcripts found under {path}")
    return [(item, load_transcript(item)) for item in files]


def load_policy(path: Path) -> PolicyPack:
    try:
        return PolicyPack.model_validate(_read_structured(path))
    except Exception as exc:
        if isinstance(exc, LoadError):
            raise
        raise LoadError(f"Invalid policy {path}: {exc}") from exc
