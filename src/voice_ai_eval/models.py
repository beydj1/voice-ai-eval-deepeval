from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TranscriptTurn(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str = Field(min_length=1)
    timestamp: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    tool_args: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    model_config = ConfigDict(extra="allow")

    call_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    expected_outcome: str | None = None
    policy: str = "default"
    turns: list[TranscriptTurn] = Field(min_length=2)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_user_and_assistant(self) -> "Transcript":
        roles = {turn.role for turn in self.turns}
        if Role.USER not in roles or Role.ASSISTANT not in roles:
            raise ValueError("Transcript must contain at least one user and one assistant turn")
        return self


class DeterministicRule(BaseModel):
    id: str
    description: str
    severity: Severity = Severity.HIGH
    type: str
    value: str | list[str] | int | float | bool | None = None


class PolicyPack(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    llm_criteria: dict[str, str] = Field(default_factory=dict)
    deterministic_rules: list[DeterministicRule] = Field(default_factory=list)


class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    severity: Severity
    message: str
    evidence: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)
    passed: bool
    reason: str
    provider: str


class EvaluationResult(BaseModel):
    call_id: str
    scenario: str
    policy_name: str
    policy_version: str
    overall_score: float = Field(ge=0, le=1)
    passed: bool
    critical_failure: bool
    deterministic_results: list[RuleResult]
    metric_results: list[MetricResult]
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_file: str | None = None

    @property
    def failed_items(self) -> list[str]:
        failures = [r.message for r in self.deterministic_results if not r.passed]
        failures.extend(m.reason for m in self.metric_results if not m.passed)
        return failures


class BatchResult(BaseModel):
    results: list[EvaluationResult]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(item.passed for item in self.results)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.overall_score for item in self.results) / len(self.results)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
