from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MetricEvaluation(BaseModel):
    """
    Structured result returned by the LLM for one evaluation criterion.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0.",
    )
    passed: bool
    reason: str = Field(
        ...,
        min_length=1,
        description="A concise explanation supporting the score.",
    )

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason must not be blank")

        return cleaned


class EvaluationResult(BaseModel):
    """
    Complete structured evaluation returned from one LLM API request.

    One transcript is evaluated against all four criteria in a single call.
    """

    task_completion: MetricEvaluation
    business_compliance: MetricEvaluation
    safety_groundedness: MetricEvaluation
    conversation_quality: MetricEvaluation