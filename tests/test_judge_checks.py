from __future__ import annotations

from typing import Any

from agent.behavior_checks import run_checks, select_judge_facts
from agent.judge_checks import (
    JudgeVerdict,
    check_expected_facts,
    check_forbidden_behaviors,
)
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

    def generate(self, prompt: str, schema: Any = None) -> JudgeVerdict:
        self.prompts.append(prompt)
        if self.error is not None:
            raise RuntimeError(self.error)
        return JudgeVerdict(passed=self.passed, explanation=self.explanation)


def make_scenario(
    facts: list[str] | None = None, forbidden: list[str] | None = None
) -> Scenario:
    return Scenario(
        id="s1",
        category="refund",
        customer_message="Can I return order 1050?",
        order_id="1050",
        expected_facts=facts if facts is not None else ["Returns use a 30-day window."],
        forbidden_behaviors=forbidden if forbidden is not None else ["Promising a refund."],
    )


def make_run(answer: str = "Yes, within 30 days.") -> AgentRun:
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


class TestExpectedFacts:
    def test_pass_verdict(self) -> None:
        judge = FakeJudge(passed=True, explanation="fact supported")

        result = check_expected_facts(make_scenario(), make_run(), judge)

        assert result.status == "pass"
        assert result.detail == "fact supported"

    def test_fail_verdict(self) -> None:
        judge = FakeJudge(passed=False, explanation="missing window detail")

        result = check_expected_facts(make_scenario(), make_run(), judge)

        assert result.status == "fail"
        assert result.detail == "missing window detail"

    def test_no_facts_skips_judge(self) -> None:
        judge = FakeJudge()

        result = check_expected_facts(make_scenario(facts=[]), make_run(), judge)

        assert result.status == "pass"
        assert result.detail == "expected facts covered by deterministic checks"
        assert judge.prompts == []

    def test_deterministic_facts_are_not_sent_to_judge(self) -> None:
        scenario = make_scenario(
            facts=[
                "Returns use a 30-day window.",
                "The agent must check the order status before cancellation.",
                "Cancellation may be attempted only if the status is processing.",
            ]
        )
        judge = FakeJudge()

        check_expected_facts(scenario, make_run(), judge)

        assert len(judge.prompts) == 1
        assert "30-day window" in judge.prompts[0]
        assert "check the order status before cancellation" not in judge.prompts[0]
        assert "only if the status is processing" not in judge.prompts[0]

    def test_judge_error_is_recorded(self) -> None:
        judge = FakeJudge(error="rate limited")

        result = check_expected_facts(make_scenario(), make_run(), judge)

        assert result.status == "error"
        assert "rate limited" in result.detail


class TestForbiddenBehaviors:
    def test_pass_when_absent(self) -> None:
        judge = FakeJudge(passed=True, explanation="no violation")

        result = check_forbidden_behaviors(make_scenario(), make_run(), judge)

        assert result.status == "pass"
        assert result.detail == "no violation"

    def test_fail_when_observed(self) -> None:
        judge = FakeJudge(passed=False, explanation="promised a refund")

        result = check_forbidden_behaviors(make_scenario(), make_run(), judge)

        assert result.status == "fail"

    def test_judge_error_is_recorded(self) -> None:
        judge = FakeJudge(error="boom")

        result = check_forbidden_behaviors(make_scenario(), make_run(), judge)

        assert result.status == "error"
        assert "boom" in result.detail


class TestRunChecksWithJudge:
    def test_judge_checks_are_appended_when_judge_present(self) -> None:
        checks = run_checks(make_scenario(), make_run(), FakeJudge())
        names = [check.name for check in checks]

        assert names == [
            "expected_tool_calls",
            "tool_arguments",
            "escalation",
            "cancellation_precondition",
            "expected_facts",
            "forbidden_behaviors",
        ]

    def test_judge_checks_omitted_without_judge(self) -> None:
        checks = run_checks(make_scenario(), make_run())
        names = [check.name for check in checks]

        assert names == [
            "expected_tool_calls",
            "tool_arguments",
            "escalation",
            "cancellation_precondition",
        ]


class TestSelectJudgeFacts:
    def test_keeps_semantic_facts_only(self) -> None:
        facts = [
            "The agent must report the actual cancellation result.",
            "The agent must check the order status before cancellation.",
            "Cancellation may be attempted only if the status is processing.",
        ]

        assert select_judge_facts(facts) == [
            "The agent must report the actual cancellation result."
        ]
