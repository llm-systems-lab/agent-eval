"""DeepEval metric evaluation for agent scenarios.

Runs the standard DeepEval metrics plus two custom metrics over each scenario's
agent answer and the evidence the agent actually used (its tool results).
Deterministic Python checks live in ``behavior_checks``; this module only adds
LLM-judge metrics. DeepEval is imported lazily so importing this module never
constructs a judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .scenarios import Scenario
from .traces import AgentRun

DEFAULT_METRIC_NAMES = (
    "Answer Relevancy",
    "Faithfulness",
    "Contextual Precision",
    "Contextual Recall",
)


class MetricScore(BaseModel):
    """Result of one DeepEval metric for a scenario."""

    name: str
    score: float | None = None
    success: bool | None = None
    reason: str | None = None
    threshold: float | None = None
    error: str | None = None


def build_retrieval_context(run: AgentRun) -> list[str]:
    """Use the agent's tool results as the evidence (retrieval context)."""
    context: list[str] = []
    for call in run.tool_calls:
        evidence = call.result if call.result is not None else call.error
        if evidence:
            context.append(f"{call.tool_name}: {evidence}")
    return context


def build_expected_output(scenario: Scenario) -> str | None:
    """Derive a reference answer from the scenario's expected facts."""
    if not scenario.expected_facts:
        return None
    return "\n".join(scenario.expected_facts)


def build_test_case(scenario: Scenario, run: AgentRun) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=scenario.customer_message,
        actual_output=run.answer,
        expected_output=build_expected_output(scenario),
        retrieval_context=build_retrieval_context(run),
        metadata={
            "scenario": scenario.model_dump(),
            "run": run.model_dump(),
        },
    )


def build_metrics(judge: Any) -> list[tuple[str, Any]]:
    """Instantiate the standard and custom DeepEval metrics bound to the judge."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    from .custom_metrics import (
        BusinessPolicyComplianceMetric,
        ToolOutcomeConsistencyMetric,
    )

    return [
        ("Answer Relevancy", AnswerRelevancyMetric(model=judge)),
        ("Faithfulness", FaithfulnessMetric(model=judge)),
        ("Contextual Precision", ContextualPrecisionMetric(model=judge)),
        ("Contextual Recall", ContextualRecallMetric(model=judge)),
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
