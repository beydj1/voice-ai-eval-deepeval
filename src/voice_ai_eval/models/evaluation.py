class MetricEvaluation(BaseModel):
    score: float
    passed: bool
    reason: str

class EvaluationResult(BaseModel):
    task_completion: MetricEvaluation
    business_compliance: MetricEvaluation
    safety_groundedness: MetricEvaluation
    conversation_quality: MetricEvaluation