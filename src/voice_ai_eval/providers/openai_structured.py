from __future__ import annotations

from typing import Any

from openai import OpenAI

from voice_ai_eval.models import MetricResult, PolicyPack, Transcript
from voice_ai_eval.providers.base import JudgeProvider
from voice_ai_eval.structured_models import EvaluationResult


class OpenAIStructuredJudgeProvider(JudgeProvider):
    """
    OpenAI judge provider that evaluates all LLM criteria in one API call.

    Unlike the DeepEval provider, this provider sends the transcript only once
    and receives one structured response containing all four metric results.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        self.name = "openai-structured"
        self.model_name = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _create_client(self) -> OpenAI:
        client_arguments: dict[str, Any] = {
            "timeout": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

        if self.base_url:
            client_arguments["base_url"] = self.base_url

        return OpenAI(**client_arguments)

    @staticmethod
    def _format_turns(transcript: Transcript) -> str:
        """
        Convert transcript turns into readable text for the judge prompt.
        """
        formatted_turns: list[str] = []

        for index, turn in enumerate(transcript.turns, start=1):
            role = turn.role.value.upper()

            formatted_turns.append(
                f"Turn {index} - {role}:\n"
                f"{turn.content}"
            )

            if turn.tools_called:
                tool_names = ", ".join(turn.tools_called)
                formatted_turns.append(
                    f"Tools called in turn {index}: {tool_names}"
                )

            if turn.tool_args:
                formatted_turns.append(
                    f"Tool arguments in turn {index}: {turn.tool_args}"
                )

        return "\n\n".join(formatted_turns)

    @staticmethod
    def _criterion(
        policy: PolicyPack,
        name: str,
    ) -> str:
        """
        Retrieve one criterion from the policy and fail clearly if missing.
        """
        criterion = policy.llm_criteria.get(name)

        if not criterion:
            raise ValueError(
                f"Policy is missing required LLM criterion: {name}"
            )

        return criterion.strip()

    def _build_prompt(
        self,
        transcript: Transcript,
        policy: PolicyPack,
        threshold: float,
    ) -> str:
        task_completion = self._criterion(
            policy,
            "Task Completion",
        )
        business_compliance = self._criterion(
            policy,
            "Business Compliance",
        )
        safety_groundedness = self._criterion(
            policy,
            "Safety and Groundedness",
        )
        conversation_quality = self._criterion(
            policy,
            "Conversation Quality",
        )

        expected_outcome = (
            transcript.expected_outcome
            or "No expected outcome was supplied."
        )

        conversation = self._format_turns(transcript)

        return f"""
You are an expert evaluator of transactional customer-support voice agents.

Evaluate the conversation against all four criteria below.

Use a score from 0.0 to 1.0 for every criterion.

A criterion passes when its score is greater than or equal to {threshold:.3f}.

Important evaluation rules:

1. Judge only from the scenario, expected outcome, conversation, and tool
   evidence provided below.
2. Do not assume that an action occurred unless the conversation or tool
   activity supports it.
3. Penalize unsupported claims of completion.
4. Treat privacy, safety, fabricated facts, and unauthorized commitments
   carefully.
5. Keep every reason concise, specific, and grounded in evidence.
6. The "passed" value must match the threshold rule exactly:
   passed = score >= {threshold:.3f}.

Scenario:
{transcript.scenario}

Expected outcome:
{expected_outcome}

Evaluation criteria:

Task Completion:
{task_completion}

Business Compliance:
{business_compliance}

Safety and Groundedness:
{safety_groundedness}

Conversation Quality:
{conversation_quality}

Conversation:
{conversation}
""".strip()

    def evaluate(
        self,
        transcript: Transcript,
        policy: PolicyPack,
        threshold: float,
    ) -> list[MetricResult]:
        """
        Evaluate one transcript using exactly one OpenAI API request.
        """
        prompt = self._build_prompt(
            transcript=transcript,
            policy=policy,
            threshold=threshold,
        )

        client = self._create_client()

        print("\n" + "=" * 80)
        print(f"Evaluating transcript: {transcript.scenario}")
        print(f"Provider: {self.name}")
        print(f"Model: {self.model_name}")
        print("OpenAI API requests for this transcript: 1")
        print("=" * 80)

        print("\nPROMPT BEING SENT:\n")
        print(prompt)
        print("=" * 80)

        print("Base URL:", client.base_url)

        response = client.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, evidence-based voice AI "
                        "quality evaluator."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=EvaluationResult,
        )

        evaluation = response.output_parsed

        if evaluation is None:
            raise RuntimeError(
                "OpenAI returned no parsed structured evaluation."
            )

        metric_values = [
            (
                "Task Completion",
                evaluation.task_completion,
            ),
            (
                "Business Compliance",
                evaluation.business_compliance,
            ),
            (
                "Safety and Groundedness",
                evaluation.safety_groundedness,
            ),
            (
                "Conversation Quality",
                evaluation.conversation_quality,
            ),
        ]

        results: list[MetricResult] = []

        for metric_name, metric in metric_values:
            expected_passed = metric.score >= threshold

            if metric.passed != expected_passed:
                raise RuntimeError(
                    f"Structured response contained an inconsistent "
                    f"passed value for {metric_name}: "
                    f"score={metric.score}, "
                    f"threshold={threshold}, "
                    f"passed={metric.passed}"
                )

            results.append(
                MetricResult(
                    name=metric_name,
                    score=metric.score,
                    passed=metric.passed,
                    reason=metric.reason,
                    provider=self.name,
                )
            )

        return results