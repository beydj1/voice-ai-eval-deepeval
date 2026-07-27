from __future__ import annotations

import os
from typing import Any

from voice_ai_eval.models import MetricResult, PolicyPack, Transcript
from voice_ai_eval.providers.base import JudgeProvider


class DeepEvalJudgeProvider(JudgeProvider):
    """DeepEval adapter supporting OpenAI and locally configured Ollama."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1",
        base_url: str | None = None,
    ) -> None:
        self.name = provider
        self.model_name = model
        self.base_url = base_url

    def _model(self) -> str | Any:
        if self.name == "ollama":
            if self.base_url:
                os.environ.setdefault(
                    "LOCAL_MODEL_BASE_URL",
                    self.base_url,
                )

            return self.model_name

        return self.model_name

    @staticmethod
    def _build_tool_calls(
        tool_names: list[str],
        tool_args: dict[str, Any],
        tool_call_class: type[Any],
    ) -> list[Any] | None:
        """
        Convert framework tool names into DeepEval ToolCall objects.
        """
        if not tool_names:
            return None

        tool_calls: list[Any] = []

        for tool_name in tool_names:
            input_parameters = tool_args.get(tool_name, {})

            if not isinstance(input_parameters, dict):
                input_parameters = {
                    "value": input_parameters,
                }

            tool_calls.append(
                tool_call_class(
                    name=tool_name,
                    input_parameters=input_parameters,
                )
            )

        return tool_calls

    def evaluate(
        self,
        transcript: Transcript,
        policy: PolicyPack,
        threshold: float,
    ) -> list[MetricResult]:
        try:
            from deepeval.metrics import ConversationalGEval
            from deepeval.test_case import (
                ConversationalTestCase,
                MultiTurnParams,
                ToolCall,
                Turn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DeepEval is not installed. Run: pip install -e '.[dev]'"
            ) from exc

        print("\n" + "=" * 80)
        print(f"Evaluating transcript: {transcript.scenario}")
        print(f"Expected outcome: {transcript.expected_outcome}")
        print(f"Turns: {len(transcript.turns)}")
        print("=" * 80)

        turns = [
            Turn(
                role=turn.role.value,
                content=turn.content,
                tools_called=self._build_tool_calls(
                    tool_names=turn.tools_called,
                    tool_args=turn.tool_args,
                    tool_call_class=ToolCall,
                ),
            )
            for turn in transcript.turns
        ]

        case = ConversationalTestCase(
            turns=turns,
            scenario=transcript.scenario,
            expected_outcome=transcript.expected_outcome,
        )

        results: list[MetricResult] = []

        for name, criteria in policy.llm_criteria.items():
            print(f"\n>>> Starting metric: {name}")

            contextual_criteria = (
                f"{criteria}\n\n"
                f"Scenario: {transcript.scenario}\n"
                f"Expected outcome: "
                f"{transcript.expected_outcome or 'Not supplied'}"
            )

            metric = ConversationalGEval(
                name=name,
                criteria=contextual_criteria,
                evaluation_params=[MultiTurnParams.CONTENT],
                threshold=threshold,
                model=self._model(),
                async_mode=False,
                verbose_mode=False,
            )

            print(f"    ✓ Metric object created: {name}")
            print(f"    → Calling metric.measure()...")

            metric.measure(case)

            print(f"    ✓ metric.measure() returned")

            score = float(metric.score or 0.0)

            print(
                f"    ✓ Score={score:.3f}, "
                f"Passed={metric.is_successful()}"
            )

            results.append(
                MetricResult(
                    name=name,
                    score=score,
                    passed=bool(metric.is_successful()),
                    reason=str(
                        metric.reason or "No reason returned"
                    ),
                    provider=self.name,
                )
            )

            print(f"<<< Finished metric: {name}")

        print("Finished transcript.")

        return results