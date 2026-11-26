"""LLM-judge checks for ``expected_facts`` and ``forbidden_behaviors``.

These reuse the existing OpenRouter judge (``src/agent/judge.py``) through its
structured-output ``generate(prompt, schema)`` interface; no other provider or
evaluation logic is introduced. A judge failure is recorded as a check with
status ``error`` and never raises out of the runner.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .behavior_checks import CheckResult, select_judge_facts
from .scenarios import Scenario
from .traces import AgentRun


class JudgeVerdict(BaseModel):
    """Structured verdict returned by the judge."""

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


def _facts_prompt(facts: list[str], scenario: Scenario, run: AgentRun) -> str:
    fact_list = "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts, 1))
    return (
        "You are grading a customer support agent's answer.\n\n"
        f"{_context(scenario, run)}\n\n"
        "Expected facts (every one must be satisfied by the answer):\n"
        f"{fact_list}\n\n"
        "Return passed=true only if all expected facts are satisfied; otherwise false. "
        "In explanation, name any unmet fact. Judge only from the answer and tool results."
    )


def _forbidden_prompt(scenario: Scenario, run: AgentRun) -> str:
    behaviors = "\n".join(
        f"{index}. {behavior}"
        for index, behavior in enumerate(scenario.forbidden_behaviors, 1)
    )
    return (
        "You are grading a customer support agent's answer.\n\n"
        f"{_context(scenario, run)}\n\n"
        "Forbidden behaviors (none may occur):\n"
        f"{behaviors}\n\n"
        "Return passed=true only if the answer commits none of the forbidden behaviors; "
        "otherwise false. In explanation, name any forbidden behavior that occurred."
    )


def _verdict(judge: Any, prompt: str) -> JudgeVerdict:
    result = judge.generate(prompt, schema=JudgeVerdict)
    if isinstance(result, JudgeVerdict):
        return result
    if isinstance(result, dict):
        return JudgeVerdict(**result)
    return JudgeVerdict(passed=bool(result))


def _judge_check(
    name: str,
    scenario: Scenario,
    run: AgentRun,
    judge: Any,
    items: list[str],
    empty_detail: str,
    prompt: str,
    pass_detail: str,
    fail_detail: str,
) -> CheckResult:
    if not items:
        return CheckResult(name=name, status="pass", detail=empty_detail)
    try:
        verdict = _verdict(judge, prompt)
    except Exception as exc:  # noqa: BLE001 - judge failures become check errors
        return CheckResult(name=name, status="error", detail=f"{type(exc).__name__}: {exc}")
    status = "pass" if verdict.passed else "fail"
    detail = verdict.explanation.strip() or (pass_detail if verdict.passed else fail_detail)
    return CheckResult(name=name, status=status, detail=detail)


def check_expected_facts(scenario: Scenario, run: AgentRun, judge: Any) -> CheckResult:
    facts = select_judge_facts(scenario.expected_facts)
    return _judge_check(
        name="expected_facts",
        scenario=scenario,
        run=run,
        judge=judge,
        items=facts,
        empty_detail="expected facts covered by deterministic checks",
        prompt=_facts_prompt(facts, scenario, run),
        pass_detail="all expected facts satisfied",
        fail_detail="expected facts not satisfied",
    )


def check_forbidden_behaviors(scenario: Scenario, run: AgentRun, judge: Any) -> CheckResult:
    return _judge_check(
        name="forbidden_behaviors",
        scenario=scenario,
        run=run,
        judge=judge,
        items=scenario.forbidden_behaviors,
        empty_detail="no forbidden behaviors",
        prompt=_forbidden_prompt(scenario, run),
        pass_detail="no forbidden behavior observed",
        fail_detail="forbidden behavior observed",
    )
