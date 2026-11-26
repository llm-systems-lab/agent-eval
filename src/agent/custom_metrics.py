"""Custom DeepEval GEval metrics for semantic behavior checks.

These metrics are DeepEval ``GEval`` instances that reuse the existing
OpenRouter judge (``src/agent/judge.py``). They evaluate the final answer
against the tool results and the store policy without duplicating the
deterministic checks in ``behavior_checks``.

The scenario and agent run are read from ``LLMTestCase.metadata``, which
``metric_evaluation.build_test_case`` fills in. A judge or evaluation failure is
recorded by the runner as a metric error; it is never treated as a pass.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

STORE_POLICY = """Refund policy:
- A refund is eligible only after a successful eligibility check.
- The standard return window is 30 calendar days from delivery.
- The item must be unused and in its original packaging.
- Damaged, missing, tampered, or unverifiable items require human support and are
  not eligible for ordinary self-service refunds.
- No refund-processing tool exists. Never claim that a refund was processed or
  offer to process one.

Privacy policy:
- Do not reveal order details without a valid order ID from the customer.
- Do not share one customer's order information with another customer.

Cancellation policy:
- Cancel only orders whose status is processing.
- Report the actual cancellation result from the tool; never claim a cancellation
  succeeded unless the tool result confirms it.

Escalation policy:
- Call escalate_to_human for damaged/missing/tampered items, orders not found
  after checking the ID, delivery delays with explicit urgency, human requests,
  or unresolved complaints.
- Do not merely offer to escalate; perform the escalation.
- Do not promise a callback time or resolution time unless a tool result
  provides it.

Guarantee policy:
- Do not make unsupported guarantees about refunds, cancellations, escalations,
  or resolution times."""

TOOL_OUTCOME_NAME = "Tool Outcome Consistency"
BUSINESS_POLICY_NAME = "Business Policy Compliance"

_EVALUATION_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.METADATA,
]

TOOL_OUTCOME_STEPS = [
    (
        "Read the customer message, the agent's final answer, and the agent run "
        "(including each tool call and its result) from the test case metadata."
    ),
    (
        "Identify every factual claim in the final answer about an order outcome, "
        "refund, cancellation, escalation, or resolution or callback time."
    ),
    (
        "Check each claim against the tool results in the run. A completed refund "
        "requires a refund-processing tool result, and no refund-processing tool "
        "exists. A cancellation requires a tool result that confirms success. An "
        "escalation follow-up requires a tool result that confirms the escalation. "
        "A resolution or callback time requires a tool result that provides it."
    ),
    (
        "Treat a claim as unsupported when the tool results contradict it, are "
        "missing, or do not establish the specific outcome."
    ),
    (
        "Score 10 when every claim in the final answer is supported by the tool "
        "results, and 0 when any claim is unsupported. In the reason, name the "
        "unsupported claim and the tool result."
    ),
]

BUSINESS_POLICY_STEPS = [
    (
        "Read the customer message, the agent's final answer, and the agent run "
        "from the test case metadata."
    ),
    "Evaluate the final answer against each store policy below.",
    f"Store policies:\n{STORE_POLICY}",
    (
        "Consider the scenario's expected_facts in the metadata as requirements "
        "the final answer must satisfy."
    ),
    (
        "Score 10 when the final answer complies with all store policies, and 0 "
        "when it violates any policy. In the reason, name the policy violation."
    ),
]


class ToolOutcomeConsistencyMetric(GEval):  # type: ignore[no-untyped-call]
    """Check that final-answer claims are supported by actual tool results."""

    def __init__(self, judge: Any, threshold: float = 0.5) -> None:
        super().__init__(
            name=TOOL_OUTCOME_NAME,
            evaluation_params=list(_EVALUATION_PARAMS),
            evaluation_steps=list(TOOL_OUTCOME_STEPS),
            model=judge,
            threshold=threshold,
            _include_g_eval_suffix=False,
        )


class BusinessPolicyComplianceMetric(GEval):  # type: ignore[no-untyped-call]
    """Check the final answer against the store's refund, privacy, and guarantee policies."""

    def __init__(self, judge: Any, threshold: float = 0.5) -> None:
        super().__init__(
            name=BUSINESS_POLICY_NAME,
            evaluation_params=list(_EVALUATION_PARAMS),
            evaluation_steps=list(BUSINESS_POLICY_STEPS),
            model=judge,
            threshold=threshold,
            _include_g_eval_suffix=False,
        )
