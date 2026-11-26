from __future__ import annotations

from typing import Any

from deepeval.models import DeepEvalBaseLLM

from agent.metric_evaluation import (
    DEFAULT_METRIC_NAMES,
    MetricScore,
    build_expected_tools,
    build_metrics,
    build_test_case,
    build_tools_called,
    build_trace,
    evaluate_scenario,
)
from agent.scenarios import Scenario
from agent.traces import AgentRun, ToolCallRecord

NEW_METRIC_NAMES = [
    "Tool Correctness",
    "Argument Correctness",
    "Task Completion",
    "Step Efficiency",
]


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


class StubJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """Minimal DeepEval model so the standard metrics can be constructed."""

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the stub judge must not be called")

    async def a_generate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the stub judge must not be called")


def make_scenario(
    facts: list[str] | None = None,
    expected_tool_calls: list[str] | None = None,
) -> Scenario:
    return Scenario(
        id="s1",
        category="refund",
        customer_message="Can I return order 1050?",
        order_id="1050",
        expected_tool_calls=(
            expected_tool_calls
            if expected_tool_calls is not None
            else ["check_refund_eligibility"]
        ),
        expected_facts=facts if facts is not None else ["Returns use a 30-day window."],
    )


def make_run(answer: str = "Yes, it is within the window.") -> AgentRun:
    return AgentRun(
        run_id="r",
        user_message="Can I return order 1050?",
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
    def test_tools_called_maps_name_arguments_and_output(self) -> None:
        tools = build_tools_called(make_run())

        assert len(tools) == 1
        assert tools[0].name == "check_refund_eligibility"
        assert tools[0].input_parameters == {"order_id": "1050"}
        assert tools[0].output == {"eligible": True}

    def test_tools_called_uses_error_when_no_result(self) -> None:
        run = AgentRun(
            run_id="r",
            user_message="m",
            model="gemini",
            tool_calls=[ToolCallRecord(tool_name="cancel_order", error="boom")],
        )

        tools = build_tools_called(run)

        assert tools[0].name == "cancel_order"
        assert tools[0].output == "boom"

    def test_tools_called_output_none_without_result_or_error(self) -> None:
        run = AgentRun(
            run_id="r",
            user_message="m",
            model="gemini",
            tool_calls=[ToolCallRecord(tool_name="get_order_status")],
        )

        assert build_tools_called(run)[0].output is None

    def test_expected_tools_maps_names(self) -> None:
        tools = build_expected_tools(make_scenario(["a"], ["get_order_status", "cancel_order"]))

        assert [tool.name for tool in tools] == ["get_order_status", "cancel_order"]

    def test_expected_tools_empty_when_none_expected(self) -> None:
        assert build_expected_tools(make_scenario(["a"], [])) == []

    def test_trace_has_agent_root_and_tool_children(self) -> None:
        trace = build_trace(make_run())

        assert trace["name"] == "customer_support_agent"
        assert trace["type"] == "agent"
        assert trace["input"] == {"input": "Can I return order 1050?"}
        assert trace["output"] == {"output": "Yes, it is within the window."}
        assert len(trace["children"]) == 1
        child = trace["children"][0]
        assert child["name"] == "check_refund_eligibility"
        assert child["type"] == "tool"
        assert child["input"] == {"inputParameters": {"order_id": "1050"}}
        assert child["output"] == {"eligible": True}
        assert child["children"] == []

    def test_trace_without_tool_calls_has_no_children(self) -> None:
        run = AgentRun(run_id="r", user_message="m", model="gemini", answer="hi")

        assert build_trace(run)["children"] == []

    def test_test_case_maps_scenario_and_run(self) -> None:
        test_case = build_test_case(make_scenario(), make_run())

        assert test_case.input == "Can I return order 1050?"
        assert test_case.actual_output == "Yes, it is within the window."
        assert [tool.name for tool in test_case.tools_called] == [
            "check_refund_eligibility"
        ]
        assert [tool.name for tool in test_case.expected_tools] == [
            "check_refund_eligibility"
        ]

    def test_test_case_includes_trace_and_metadata(self) -> None:
        test_case = build_test_case(make_scenario(), make_run())

        assert test_case._trace_dict["name"] == "customer_support_agent"
        assert test_case.metadata["scenario"]["id"] == "s1"
        assert test_case.metadata["run"]["answer"] == "Yes, it is within the window."


class TestBuildMetrics:
    def test_default_metric_names_are_the_agent_metrics(self) -> None:
        assert list(DEFAULT_METRIC_NAMES) == NEW_METRIC_NAMES

    def test_build_metrics_returns_agent_and_custom_metrics(self) -> None:
        judge = StubJudge()

        metrics = build_metrics(judge)

        assert [name for name, _ in metrics] == NEW_METRIC_NAMES + [
            "Tool Outcome Consistency",
            "Business Policy Compliance",
        ]

    def test_standard_metrics_are_bound_to_the_judge(self) -> None:
        judge = StubJudge()

        metrics = dict(build_metrics(judge))

        for name in NEW_METRIC_NAMES:
            assert metrics[name].model is judge
            assert metrics[name].__name__ == name

    def test_custom_metrics_are_bound_to_the_judge(self) -> None:
        judge = StubJudge()

        metrics = dict(build_metrics(judge))

        assert metrics["Tool Outcome Consistency"].model is judge
        assert metrics["Business Policy Compliance"].model is judge
        assert metrics["Tool Outcome Consistency"].name == "Tool Outcome Consistency"
        assert (
            metrics["Business Policy Compliance"].name == "Business Policy Compliance"
        )


class TestEvaluateScenario:
    def test_records_scores_for_each_metric(self) -> None:
        metrics = [
            ("Tool Correctness", FakeMetric(score=0.8, success=True, reason="ok")),
            ("Argument Correctness", FakeMetric(score=1.0, success=True, reason="exact")),
        ]

        scores = evaluate_scenario(make_scenario(), make_run(), metrics)

        assert [s.name for s in scores] == ["Tool Correctness", "Argument Correctness"]
        assert scores[0].score == 0.8
        assert scores[1].score == 1.0
        assert scores[0].threshold == 0.5
        assert all(s.error is None for s in scores)

    def test_metric_error_is_recorded_and_execution_continues(self) -> None:
        metrics = [
            ("Tool Correctness", FakeMetric(error="rate limited")),
            ("Task Completion", FakeMetric(score=0.7)),
        ]

        scores = evaluate_scenario(make_scenario(), make_run(), metrics)

        assert scores[0].error is not None
        assert "rate limited" in scores[0].error
        assert scores[1].score == 0.7

    def test_metrics_measure_the_built_test_case(self) -> None:
        metric = FakeMetric()

        evaluate_scenario(make_scenario(), make_run(), [("Step Efficiency", metric)])

        assert metric.measured_test_case.input == "Can I return order 1050?"
        assert metric.measured_test_case._trace_dict["name"] == "customer_support_agent"


def test_metric_score_defaults() -> None:
    score = MetricScore(name="Tool Correctness")

    assert score.score is None
    assert score.success is None
    assert score.reason is None
    assert score.threshold is None
    assert score.error is None
