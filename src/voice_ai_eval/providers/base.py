from __future__ import annotations

from abc import ABC, abstractmethod

from voice_ai_eval.models import MetricResult, PolicyPack, Transcript


class JudgeProvider(ABC):
    name: str

    @abstractmethod
    def evaluate(self, transcript: Transcript, policy: PolicyPack, threshold: float) -> list[MetricResult]:
        raise NotImplementedError
