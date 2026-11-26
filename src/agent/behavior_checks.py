"""Checks over structured agent traces.

Expected tool calls, tool arguments, and escalation are deterministic Python
checks. ``expected_facts`` and ``forbidden_behaviors`` are judged by the LLM
judge (see ``judge_checks``); when no judge is supplied those two checks are
omitted rather than guessed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel

from .scenarios import Scenario
from .traces import AgentRun

ORDER_SPECIFIC_TOOLS = frozenset(
    {
        "get_order_status",
        "get_order_details",
        "check_refund_eligibility",
        "cancel_order",
    }
)

TOOL_NAMES = (
    "get_order_status",
    "get_order_details",
    "check_refund_eligibility",
    "cancel_order",
    "escalate_to_human",
)

# Facts that describe rules the deterministic checks already verify, so the LLM
# judge is not asked to (re-)judge them.
DETERMINISTIC_FACT_PATTERNS = (
    re.compile(r"check the order status before cancel", re.IGNORECASE),
    re.compile(r"only if the status is processing", re.IGNORECASE),
)

CheckStatus = Literal["pass", "fail", "error"]


class CheckResult(BaseModel):
    """Outcome of one deterministic behavior check."""

    name: str
    status: CheckStatus
    detail: str = ""


def _normalize_order_id(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lstrip("#").strip()
    return normalized or None


def _called_names(run: AgentRun) -> list[str]:
    return [call.tool_name for call in run.tool_calls]


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(item in iterator for item in expected)


def _escalation_succeeded(run: AgentRun) -> bool:
    for call in run.tool_calls:
        if call.tool_name != "escalate_to_human" or call.result is None:
            continue
        try:
            payload = json.loads(call.result)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("success") is True:
            return True
    return False


def _forbidden_tools(scenario: Scenario) -> set[str]:
    """Tool names that appear in the forbidden-behavior text.

    A tool named in ``expected_tool_calls`` is never treated as forbidden, so
    ``expected_tool_calls`` and ``forbidden_tool_calls`` stay consistent even
    when a forbidden-behavior sentence mentions an expected tool conditionally
    (for example "Calling cancel_order before checking the status").
    """
    text = " ".join(scenario.forbidden_behaviors)
    mentioned = {name for name in TOOL_NAMES if name in text}
    return mentioned - set(scenario.expected_tool_calls)


def is_deterministic_fact(fact: str) -> bool:
    """Whether a scenario fact is already covered by a deterministic check."""
    return any(pattern.search(fact) for pattern in DETERMINISTIC_FACT_PATTERNS)


def select_judge_facts(facts: list[str]) -> list[str]:
    """Facts that still need the LLM judge (semantic, not deterministic)."""
    return [fact for fact in facts if not is_deterministic_fact(fact)]


def check_expected_tool_calls(scenario: Scenario, run: AgentRun) -> CheckResult:
    expected = scenario.expected_tool_calls
    actual = _called_names(run)
    forbidden = _forbidden_tools(scenario)
    offending = [name for name in actual if name in forbidden]
    if offending:
        return CheckResult(
            name="expected_tool_calls",
            status="fail",
            detail=f"called forbidden tool(s) {offending}; actual {actual}",
        )

    extras = [name for name in actual if name not in expected]

    if not expected:
        if not actual:
            return CheckResult(
                name="expected_tool_calls",
                status="pass",
                detail="no tool calls expected",
            )
        return CheckResult(
            name="expected_tool_calls",
            status="fail",
            detail=f"no tool calls expected but the agent called {actual}",
        )

    missing = [name for name in expected if name not in actual]
    if missing:
        return CheckResult(
            name="expected_tool_calls",
            status="fail",
            detail=f"missing expected tool calls {missing}; actual {actual}",
        )

    if not _is_subsequence(expected, actual):
        return CheckResult(
            name="expected_tool_calls",
            status="fail",
            detail=f"expected order {expected} not preserved in actual {actual}",
        )

    detail = f"all expected tool calls present in order {expected}"
    if extras:
        detail += f"; extra calls {extras}"
    return CheckResult(name="expected_tool_calls", status="pass", detail=detail)


def check_tool_arguments(scenario: Scenario, run: AgentRun) -> CheckResult:
    if scenario.order_id is None:
        return CheckResult(
            name="tool_arguments",
            status="pass",
            detail="scenario has no order ID to check",
        )

    expected_id = _normalize_order_id(scenario.order_id)
    checked = 0
    mismatches: list[str] = []
    for call in run.tool_calls:
        if call.tool_name not in ORDER_SPECIFIC_TOOLS:
            continue
        checked += 1
        actual_id = _normalize_order_id(call.arguments.get("order_id"))
        if actual_id != expected_id:
            mismatches.append(
                f"{call.tool_name}(order_id={call.arguments.get('order_id')!r})"
            )

    if mismatches:
        return CheckResult(
            name="tool_arguments",
            status="fail",
            detail=f"order ID mismatch {mismatches}; expected {expected_id}",
        )
    if checked == 0:
        return CheckResult(
            name="tool_arguments",
            status="pass",
            detail="no order-specific tool calls were made",
        )
    return CheckResult(
        name="tool_arguments",
        status="pass",
        detail=f"{checked} order-specific call(s) used order {expected_id}",
    )


def check_escalation(scenario: Scenario, run: AgentRun) -> CheckResult:
    actual = _escalation_succeeded(run)
    return CheckResult(
        name="escalation",
        status="pass" if actual == scenario.should_escalate else "fail",
        detail=f"escalation expected={scenario.should_escalate}, actual={actual}",
    )


def check_cancellation_precondition(scenario: Scenario, run: AgentRun) -> CheckResult:
    """A cancellation requires a preceding status check that returned 'processing'."""
    cancel_index = next(
        (
            index
            for index, call in enumerate(run.tool_calls)
            if call.tool_name == "cancel_order"
        ),
        None,
    )
    if cancel_index is None:
        return CheckResult(
            name="cancellation_precondition",
            status="pass",
            detail="cancel_order was not called",
        )

    for call in run.tool_calls[:cancel_index]:
        if call.tool_name != "get_order_status" or call.result is None:
            continue
        try:
            payload = json.loads(call.result)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("status") == "processing":
            return CheckResult(
                name="cancellation_precondition",
                status="pass",
                detail="get_order_status returned 'processing' before cancel_order",
            )
    return CheckResult(
        name="cancellation_precondition",
        status="fail",
        detail="cancel_order was called without a preceding get_order_status "
        "that returned 'processing'",
    )


def run_checks(
    scenario: Scenario,
    run: AgentRun,
    judge: Any | None = None,
) -> list[CheckResult]:
    """Run the checks for one scenario against its trace.

    Expected tool calls, tool arguments, escalation, and the cancellation
    precondition are always checked deterministically. When ``judge`` is
    provided, semantic ``expected_facts`` and ``forbidden_behaviors`` are
    additionally checked by the LLM judge.
    """
    checks = [
        check_expected_tool_calls(scenario, run),
        check_tool_arguments(scenario, run),
        check_escalation(scenario, run),
        check_cancellation_precondition(scenario, run),
    ]
    if judge is not None:
        from .judge_checks import (
            check_expected_facts,
            check_forbidden_behaviors,
        )

        checks.append(check_expected_facts(scenario, run, judge))
        checks.append(check_forbidden_behaviors(scenario, run, judge))
    return checks
