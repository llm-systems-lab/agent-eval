from __future__ import annotations

from typing import Any

from agent.metric_evaluation import (
    MetricScore,
    build_expected_output,
    build_retrieval_context,
    build_test_case,
    evaluate_scenario,
)
from agent.scenarios import Scenario
from agent.traces import AgentRun, ToolCallRecord


class FakeMetric:
    """Stand-in DeepEval metric that records the test case it measured."""

    def __init__(
        self,
        score: float = 0.9,
        success: bool = True,
        reason: str = "good",
        error: str | None = None,
        threshold: float | None = 0.5,
    ) -> None:
        self.score = score
        self.success = success
        self.reason = reason
        self.error = error
        self.threshold = threshold
        self.measured_test_case: Any = None

    def measure(self, test_case: Any) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)
        self.measured_test_case = test_case


def make_scenario(facts: list[str] | None = None) -> Scenario:
    return Scenario(
        id="s1",
        category="refund",
        customer_message="Can I return order 1050?",
        order_id="1050",
        expected_facts=facts if facts is not None else ["Returns use a 30-day window."],
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


class TestBuilders:
    def test_retrieval_context_uses_tool_results(self) -> None:
        context = build_retrieval_context(make_run())

        assert context == ['check_refund_eligibility: {"eligible": true}']

    def test_retrieval_context_uses_error_when_no_result(self) -> None:
        run = AgentRun(
            run_id="r",
            user_message="m",
            model="gemini",
            tool_calls=[ToolCallRecord(tool_name="cancel_order", error="boom")],
        )

        assert build_retrieval_context(run) == ["cancel_order: boom"]

    def test_expected_output_joins_facts(self) -> None:
        assert build_expected_output(make_scenario(["a", "b"])) == "a\nb"

    def test_expected_output_none_when_no_facts(self) -> None:
        assert build_expected_output(make_scenario([])) is None

    def test_test_case_maps_scenario_and_run(self) -> None:
        test_case = build_test_case(make_scenario(), make_run())

        assert test_case.input == "Can I return order 1050?"
        assert test_case.actual_output == "Yes, it is within the window."
        assert test_case.expected_output == "Returns use a 30-day window."
        assert test_case.retrieval_context == ['check_refund_eligibility: {"eligible": true}']

    def test_test_case_includes_metadata_evidence(self) -> None:
        test_case = build_test_case(make_scenario(), make_run())

        assert test_case.metadata["scenario"]["id"] == "s1"
        assert test_case.metadata["run"]["answer"] == "Yes, it is within the window."


class TestEvaluateScenario:
    def test_records_scores_for_each_metric(self) -> None:
        metrics = [
            ("Answer Relevancy", FakeMetric(score=0.8, success=True, reason="ok")),
            ("Faithfulness", FakeMetric(score=1.0, success=True, reason="grounded")),
        ]

        scores = evaluate_scenario(make_scenario(), make_run(), metrics)

        assert [s.name for s in scores] == ["Answer Relevancy", "Faithfulness"]
        assert scores[0].score == 0.8
        assert scores[1].score == 1.0
        assert scores[0].threshold == 0.5
        assert all(s.error is None for s in scores)

    def test_metric_error_is_recorded_and_execution_continues(self) -> None:
        metrics = [
            ("Answer Relevancy", FakeMetric(error="rate limited")),
            ("Faithfulness", FakeMetric(score=0.7)),
        ]

        scores = evaluate_scenario(make_scenario(), make_run(), metrics)

        assert scores[0].error is not None
        assert "rate limited" in scores[0].error
        assert scores[1].score == 0.7

    def test_metrics_measure_the_built_test_case(self) -> None:
        metric = FakeMetric()

        evaluate_scenario(make_scenario(), make_run(), [("Answer Relevancy", metric)])

        assert metric.measured_test_case.input == "Can I return order 1050?"


def test_metric_score_defaults() -> None:
    score = MetricScore(name="Answer Relevancy")

    assert score.score is None
    assert score.success is None
    assert score.reason is None
    assert score.threshold is None
    assert score.error is None
