from __future__ import annotations

from typing import Any

from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.models import DeepEvalBaseLLM

from agent.custom_metrics import (
    BusinessPolicyComplianceMetric,
    ToolOutcomeConsistencyMetric,
)
from agent.metric_evaluation import build_test_case, evaluate_scenario
from agent.scenarios import Scenario
from agent.traces import AgentRun, ToolCallRecord


class StubJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """GEval judge stub returning a scripted reason/score or raising."""

    def __init__(
        self,
        score: float = 10.0,
        reason: str = "looks fine",
        error: str | None = None,
    ) -> None:
        self._score = score
        self._reason = reason
        self._error = error
        self.prompts: list[str] = []

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self._error is not None:
            raise RuntimeError(self._error)
        return ReasonScore(reason=self._reason, score=self._score)

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        return self.generate(prompt, *args, **kwargs)


def make_scenario() -> Scenario:
    return Scenario(
        id="s1",
        category="refund",
        customer_message="Can I return order 1050?",
        order_id="1050",
        expected_facts=["Returns use a 30-day window."],
    )


def make_run(answer: str = "Yes, it is within the window.") -> AgentRun:
    return AgentRun(
        run_id="r",
        user_message="m",
        model="gemini",
        answer=answer,
        tool_calls=[
            ToolCallRecord(
                tool_name="check_refund_eligibility",
                arguments={"order_id": "1050"},
                result='{"eligible": true}',
            )
        ],
    )


def make_test_case() -> Any:
    return build_test_case(make_scenario(), make_run())


class TestToolOutcomeConsistencyMetric:
    def test_pass(self) -> None:
        judge = StubJudge(score=10.0, reason="supported")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.score == 1.0
        assert metric.success is True
        assert metric.reason == "supported"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = StubJudge(score=0.0, reason="claimed a refund")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "claimed a refund"

    def test_prompt_contains_evidence(self) -> None:
        judge = StubJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Can I return order 1050?" in prompt
        assert "Yes, it is within the window." in prompt
        assert "check_refund_eligibility" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_name_and_threshold(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge())

        assert metric.name == "Tool Outcome Consistency"
        assert metric.threshold == 0.5


class TestBusinessPolicyComplianceMetric:
    def test_pass(self) -> None:
        judge = StubJudge(score=10.0, reason="compliant")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.success is True
        assert metric.reason == "compliant"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = StubJudge(score=0.0, reason="promised a refund")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "promised a refund"

    def test_prompt_contains_policy(self) -> None:
        judge = StubJudge()
        metric = BusinessPolicyComplianceMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Refund policy" in prompt
        assert "Privacy policy" in prompt
        assert "Guarantee policy" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_name_and_threshold(self) -> None:
        metric = BusinessPolicyComplianceMetric(StubJudge())

        assert metric.name == "Business Policy Compliance"
        assert metric.threshold == 0.5


class TestEvaluateScenarioIntegration:
    def test_custom_metric_pass_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge(score=10.0))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 1.0
        assert scores[0].success is True
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_fail_recorded(self) -> None:
        metric = BusinessPolicyComplianceMetric(StubJudge(score=0.0))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Business Policy Compliance", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 0.0
        assert scores[0].success is False
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_error_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge(error="boom"))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].error is not None
        assert "boom" in scores[0].error
        assert scores[0].success is None
