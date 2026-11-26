from __future__ import annotations

from typing import Any

import pytest

from agent.custom_metrics import (
    BusinessPolicyComplianceMetric,
    ToolOutcomeConsistencyMetric,
)
from agent.metric_evaluation import build_test_case, evaluate_scenario
from agent.scenarios import Scenario
from agent.traces import AgentRun, ToolCallRecord


class FakeJudge:
    """Judge stub returning a scripted verdict or raising."""

    def __init__(
        self,
        passed: bool = True,
        explanation: str = "looks fine",
        error: str | None = None,
    ) -> None:
        self.passed = passed
        self.explanation = explanation
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: Any = None) -> dict[str, Any]:
        self.prompts.append(prompt)
        if self.error is not None:
            raise RuntimeError(self.error)
        return {"passed": self.passed, "explanation": self.explanation}


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
        judge = FakeJudge(passed=True, explanation="supported")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.score == 1.0
        assert metric.success is True
        assert metric.reason == "supported"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = FakeJudge(passed=False, explanation="claimed a refund")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "claimed a refund"
        assert metric.error is None

    def test_error(self) -> None:
        judge = FakeJudge(error="rate limited")
        metric = ToolOutcomeConsistencyMetric(judge)

        with pytest.raises(RuntimeError):
            metric.measure(make_test_case())

        assert metric.score is None
        assert metric.success is False
        assert metric.error is not None
        assert "rate limited" in metric.error

    def test_missing_metadata_is_error(self) -> None:
        from deepeval.test_case import LLMTestCase

        metric = ToolOutcomeConsistencyMetric(FakeJudge())

        with pytest.raises(ValueError):
            metric.measure(LLMTestCase(input="x", actual_output="y"))

        assert metric.error is not None
        assert metric.success is False

    def test_prompt_contains_evidence(self) -> None:
        judge = FakeJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Can I return order 1050?" in prompt
        assert "Yes, it is within the window." in prompt
        assert "check_refund_eligibility" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_name_and_threshold(self) -> None:
        metric = ToolOutcomeConsistencyMetric(FakeJudge())

        assert metric.name == "Tool Outcome Consistency"
        assert metric.threshold == 0.5


class TestBusinessPolicyComplianceMetric:
    def test_pass(self) -> None:
        judge = FakeJudge(passed=True, explanation="compliant")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.success is True
        assert metric.reason == "compliant"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = FakeJudge(passed=False, explanation="promised a refund")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "promised a refund"
        assert metric.error is None

    def test_error(self) -> None:
        judge = FakeJudge(error="judge down")
        metric = BusinessPolicyComplianceMetric(judge)

        with pytest.raises(RuntimeError):
            metric.measure(make_test_case())

        assert metric.score is None
        assert metric.success is False
        assert metric.error is not None
        assert "judge down" in metric.error

    def test_prompt_contains_policy(self) -> None:
        judge = FakeJudge()
        metric = BusinessPolicyComplianceMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Refund policy" in prompt
        assert "Privacy policy" in prompt
        assert "Guarantee policy" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_name_and_threshold(self) -> None:
        metric = BusinessPolicyComplianceMetric(FakeJudge())

        assert metric.name == "Business Policy Compliance"
        assert metric.threshold == 0.5


class TestEvaluateScenarioIntegration:
    def test_custom_metric_pass_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(FakeJudge(passed=True))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 1.0
        assert scores[0].success is True
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_fail_recorded(self) -> None:
        metric = BusinessPolicyComplianceMetric(FakeJudge(passed=False))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Business Policy Compliance", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 0.0
        assert scores[0].success is False
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_error_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(FakeJudge(error="boom"))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].error is not None
        assert "boom" in scores[0].error
        assert scores[0].success is None
