"""DeepEval agent metrics for evaluation scenarios.

Runs the standard DeepEval agent metrics plus two custom metrics over each
scenario's agent answer, the tools it called, and its structured execution
trace. Deterministic Python checks live in ``behavior_checks``; this module only
adds LLM-judge metrics. DeepEval is imported lazily so importing this module
never constructs a judge.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .scenarios import Scenario
from .traces import AgentRun, ToolCallRecord

DEFAULT_METRIC_NAMES = (
    "Tool Correctness",
    "Argument Correctness",
    "Task Completion",
    "Step Efficiency",
)

AGENT_TRACE_NAME = "customer_support_agent"


class MetricScore(BaseModel):
    """Result of one DeepEval metric for a scenario."""

    name: str
    score: float | None = None
    success: bool | None = None
    reason: str | None = None
    threshold: float | None = None
    error: str | None = None


def _tool_output(call: ToolCallRecord) -> Any:
    """Return a tool call's result, parsed from JSON when possible."""
    raw = call.result if call.result is not None else call.error
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def build_tools_called(run: AgentRun) -> list[Any]:
    """Map the agent's executed tool calls to DeepEval ``ToolCall`` objects."""
    from deepeval.test_case import ToolCall

    return [
        ToolCall(
            name=call.tool_name,
            input_parameters=dict(call.arguments),
            output=_tool_output(call),
        )
        for call in run.tool_calls
    ]


def build_expected_tools(scenario: Scenario) -> list[Any]:
    """Map the scenario's expected tool names to DeepEval ``ToolCall`` objects."""
    from deepeval.test_case import ToolCall

    return [
        ToolCall(name=name, input_parameters=None)
        for name in scenario.expected_tool_calls
    ]


def build_trace(run: AgentRun) -> dict[str, Any]:
    """Build a nested span trace from the agent's structured execution data."""
    children: list[dict[str, Any]] = []
    for call in run.tool_calls:
        children.append(
            {
                "name": call.tool_name,
                "type": "tool",
                "input": {"inputParameters": dict(call.arguments)},
                "output": _tool_output(call),
                "children": [],
            }
        )
    return {
        "name": AGENT_TRACE_NAME,
        "type": "agent",
        "input": {"input": run.user_message},
        "output": {"output": run.answer},
        "children": children,
    }


def build_test_case(scenario: Scenario, run: AgentRun) -> Any:
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=scenario.customer_message,
        actual_output=run.answer,
        tools_called=build_tools_called(run),
        expected_tools=build_expected_tools(scenario),
        metadata={
            "scenario": scenario.model_dump(),
            "run": run.model_dump(),
        },
    )
    test_case._trace_dict = build_trace(run)
    return test_case


def build_metrics(judge: Any) -> list[tuple[str, Any]]:
    """Instantiate the standard and custom DeepEval metrics bound to the judge."""
    from deepeval.metrics import ToolCorrectnessMetric

    from .custom_metrics import (
        BusinessPolicyComplianceMetric,
        ToolOutcomeConsistencyMetric,
        policy_aware_argument_correctness_metric,
        policy_aware_step_efficiency_metric,
        policy_aware_task_completion_metric,
    )

    return [
        ("Tool Correctness", ToolCorrectnessMetric(model=judge, async_mode=False)),
        ("Argument Correctness", policy_aware_argument_correctness_metric(judge)),
        ("Task Completion", policy_aware_task_completion_metric(judge)),
        ("Step Efficiency", policy_aware_step_efficiency_metric(judge)),
        ("Tool Outcome Consistency", ToolOutcomeConsistencyMetric(judge)),
        ("Business Policy Compliance", BusinessPolicyComplianceMetric(judge)),
    ]


def evaluate_scenario(
    scenario: Scenario,
    run: AgentRun,
    metrics: Sequence[tuple[str, Any]],
) -> list[MetricScore]:
    """Measure each metric, recording failures instead of raising."""
    test_case = build_test_case(scenario, run)
    scores: list[MetricScore] = []
    for name, metric in metrics:
        try:
            metric.measure(test_case)
            scores.append(
                MetricScore(
                    name=name,
                    score=getattr(metric, "score", None),
                    success=getattr(metric, "success", None),
                    reason=getattr(metric, "reason", None),
                    threshold=getattr(metric, "threshold", None),
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            scores.append(
                MetricScore(
                    name=name,
                    threshold=getattr(metric, "threshold", None),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return scores
