"""Custom DeepEval metrics for semantic behavior checks.

These metrics extend ``deepeval.metrics.BaseMetric`` and reuse the existing
OpenRouter judge (``src/agent/judge.py``) through its structured-output
``generate(prompt, schema)`` interface. They evaluate the final answer against
the tool results and the store policy without duplicating the deterministic
checks in ``behavior_checks``.

The metrics read the scenario and agent run from the ``LLMTestCase.metadata``
field, which ``metric_evaluation.build_test_case`` fills in. A judge or
evaluation failure is stored on the metric and re-raised so the runner records
it as an error; it is never treated as a pass.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import BaseMetric
from pydantic import BaseModel

from .scenarios import Scenario
from .traces import AgentRun

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


class ToolOutcomeVerdict(BaseModel):
    """Structured verdict for tool-outcome consistency."""

    passed: bool
    explanation: str = ""


class PolicyVerdict(BaseModel):
    """Structured verdict for business-policy compliance."""

    passed: bool
    explanation: str = ""


def _evidence(run: AgentRun) -> str:
    lines: list[str] = []
    for call in run.tool_calls:
        outcome = call.result if call.result is not None else call.error
        lines.append(f"- {call.tool_name}({call.arguments}) -> {outcome}")
    return "\n".join(lines) or "(no tool calls)"


def _context(scenario: Scenario, run: AgentRun) -> str:
    return (
        f"Customer message:\n{scenario.customer_message}\n\n"
        f"Agent answer:\n{run.answer}\n\n"
        f"Tool results:\n{_evidence(run)}"
    )


def _facts(scenario: Scenario) -> str:
    return "\n".join(f"- {fact}" for fact in scenario.expected_facts) or "(none)"


def _extract_evidence(test_case: Any) -> tuple[Scenario, AgentRun]:
    """Recover the scenario and agent run from the test case metadata."""
    metadata = getattr(test_case, "metadata", None) or {}
    scenario_data = metadata.get("scenario")
    run_data = metadata.get("run")
    if not scenario_data or not run_data:
        raise ValueError("test case metadata must contain 'scenario' and 'run'")
    return Scenario.model_validate(scenario_data), AgentRun.model_validate(run_data)


def _judge_verdict(
    judge: Any, prompt: str, schema: type[BaseModel]
) -> BaseModel:
    result = judge.generate(prompt, schema=schema)
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema(**result)
    return schema(passed=bool(result))


class _BaseCustomMetric(BaseMetric):
    """Shared plumbing for the custom judge-backed metrics."""

    _name = "Custom Metric"
    _pass_reason = "consistent"
    _fail_reason = "inconsistent"

    def __init__(self, judge: Any, threshold: float = 0.5) -> None:
        self._judge = judge
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def __name__(self) -> str:
        return self._name

    def _verdict_model(self) -> type[BaseModel]:
        raise NotImplementedError

    def _prompt(self, scenario: Scenario, run: AgentRun) -> str:
        raise NotImplementedError

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        try:
            scenario, run = _extract_evidence(test_case)
            verdict = _judge_verdict(
                self._judge, self._prompt(scenario, run), self._verdict_model()
            )
            self.score = 1.0 if verdict.passed else 0.0
            self.reason = verdict.explanation.strip() or (
                self._pass_reason if verdict.passed else self._fail_reason
            )
            self.success = self.score >= self.threshold
        except Exception as exc:  # record and re-raise for the runner
            self.error = f"{type(exc).__name__}: {exc}"
            self.success = False
            raise
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)


class ToolOutcomeConsistencyMetric(_BaseCustomMetric):
    """Check that final-answer claims are supported by actual tool results."""

    _name = "Tool Outcome Consistency"
    _pass_reason = "all claims supported by tool results"
    _fail_reason = "unsupported claim in final answer"

    def _verdict_model(self) -> type[BaseModel]:
        return ToolOutcomeVerdict

    def _prompt(self, scenario: Scenario, run: AgentRun) -> str:
        return (
            "You are grading a customer support agent's final answer for consistency "
            "with the tool execution results.\n\n"
            f"{_context(scenario, run)}\n\n"
            "Relevant expected facts:\n"
            f"{_facts(scenario)}\n\n"
            "Check whether the final answer makes any claim that is not supported by "
            "the tool results. Pay special attention to claims about:\n"
            "- completed refunds (no refund-processing tool exists)\n"
            "- cancellations (the tool result must confirm success)\n"
            "- escalation follow-ups (the tool result must confirm the escalation)\n"
            "- resolution times or callback times (the tool result must provide them)\n\n"
            "A claim is unsupported if the tool results contradict it, are missing, or "
            "do not establish the specific outcome. Return passed=true only if every "
            "claim in the final answer is supported by the tool results; otherwise "
            "false. In explanation, name the unsupported claim and the tool result."
        )


class BusinessPolicyComplianceMetric(_BaseCustomMetric):
    """Check the final answer against the store's refund, privacy, and guarantee policies."""

    _name = "Business Policy Compliance"
    _pass_reason = "answer complies with store policy"
    _fail_reason = "answer violates store policy"

    def _verdict_model(self) -> type[BaseModel]:
        return PolicyVerdict

    def _prompt(self, scenario: Scenario, run: AgentRun) -> str:
        return (
            "You are grading a customer support agent's final answer for compliance "
            "with the store's policies.\n\n"
            f"Store policies:\n{STORE_POLICY}\n\n"
            f"{_context(scenario, run)}\n\n"
            "Relevant expected facts:\n"
            f"{_facts(scenario)}\n\n"
            "Return passed=true only if the final answer complies with all store "
            "policies; otherwise false. In explanation, name the policy violation."
        )
