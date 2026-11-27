from __future__ import annotations

from typing import Any

from deepeval.metrics.argument_correctness.schema import (
    ArgumentCorrectnessScoreReason,
    ArgumentCorrectnessVerdict,
    Verdicts,
)
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.metrics.step_efficiency.schema import EfficiencyVerdict, Task
from deepeval.metrics.task_completion.schema import TaskCompletionVerdict
from deepeval.models import DeepEvalBaseLLM

from agent.custom_metrics import (
    BusinessPolicyComplianceMetric,
    PolicyAwareArgumentCorrectnessTemplate,
    PolicyAwareStepEfficiencyTemplate,
    PolicyAwareTaskCompletionTemplate,
    ToolOutcomeConsistencyMetric,
    policy_aware_argument_correctness_metric,
    policy_aware_step_efficiency_metric,
    policy_aware_task_completion_metric,
)
from agent.metric_evaluation import build_metrics, build_test_case, evaluate_scenario
from agent.scenarios import Scenario
from agent.traces import AgentRun, ToolCallRecord


class StubJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """GEval judge stub returning a scripted reason/score or raising."""

    def __init__(
        self,
        score: float = 10.0,
        reason: str = "looks fine",
        error: str | None = None,
    ) -> None:
        self._score = score
        self._reason = reason
        self._error = error
        self.prompts: list[str] = []

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self._error is not None:
            raise RuntimeError(self._error)
        return ReasonScore(reason=self._reason, score=self._score)

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        return self.generate(prompt, *args, **kwargs)


class TaskCompletionJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """Task Completion judge stub returning a scripted verdict and capturing prompts."""

    def __init__(self, verdict: float = 1.0, reason: str = "ok") -> None:
        self._verdict = verdict
        self._reason = reason
        self.prompts: list[str] = []

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, schema: Any = None) -> Any:
        self.prompts.append(prompt)
        if schema is TaskCompletionVerdict:
            return TaskCompletionVerdict(verdict=self._verdict, reason=self._reason)
        raise AssertionError(f"unexpected schema: {schema}")

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)


class StepEfficiencyJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """Step Efficiency judge stub returning a scripted verdict and capturing prompts."""

    def __init__(self, score: float = 1.0, reason: str = "ok") -> None:
        self._score = score
        self._reason = reason
        self.prompts: list[str] = []

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, schema: Any = None) -> Any:
        self.prompts.append(prompt)
        if schema is Task:
            return Task(task="resolve the customer request")
        if schema is EfficiencyVerdict:
            return EfficiencyVerdict(score=self._score, reason=self._reason)
        raise AssertionError(f"unexpected schema: {schema}")

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)


class ArgumentCorrectnessJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """Argument Correctness judge stub returning a scripted verdict."""

    def __init__(
        self,
        verdict: Any = Verdict.YES,
        reason: str = "ok",
    ) -> None:
        self._verdict = verdict
        self._reason = reason
        self.prompts: list[str] = []

    def load_model(self) -> Any:
        return None

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, schema: Any = None) -> Any:
        self.prompts.append(prompt)
        if schema is Verdicts:
            return Verdicts(
                verdicts=[ArgumentCorrectnessVerdict(verdict=self._verdict)]
            )
        if schema is ArgumentCorrectnessScoreReason:
            return ArgumentCorrectnessScoreReason(reason=self._reason)
        raise AssertionError(f"unexpected schema: {schema}")

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)


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
        judge = StubJudge(score=10.0, reason="supported")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.score == 1.0
        assert metric.success is True
        assert metric.reason == "supported"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = StubJudge(score=0.0, reason="claimed a refund")
        metric = ToolOutcomeConsistencyMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "claimed a refund"

    def test_prompt_contains_evidence(self) -> None:
        judge = StubJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Can I return order 1050?" in prompt
        assert "Yes, it is within the window." in prompt
        assert "check_refund_eligibility" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_prompt_treats_absent_detail_as_supported(self) -> None:
        judge = StubJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = " ".join(judge.prompts[0].lower().split())
        assert (
            "is supported when the recorded tool results do not contain that detail"
            in prompt
        )
        assert "do not treat a truthful 'not available' as unsupported" in prompt
        assert (
            "tool explicitly provided or confirmed information it did not" in prompt
        )

    def test_prompt_allows_business_policy_support(self) -> None:
        judge = StubJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = " ".join(judge.prompts[0].lower().split())
        assert "cancel only orders whose status is processing" in prompt
        assert "supported by the cancellation policy" in prompt
        assert "require tool evidence for claims about completed actions" in prompt

    def test_prompt_treats_customer_reported_facts_as_supported(self) -> None:
        judge = StubJudge()
        metric = ToolOutcomeConsistencyMetric(judge)

        metric.measure(make_test_case())

        prompt = " ".join(judge.prompts[0].lower().split())
        assert "the customer themselves stated" in prompt
        assert "does not require tool verification" in prompt
        assert "customer's own description" in prompt

    def test_name_and_threshold(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge())

        assert metric.name == "Tool Outcome Consistency"
        assert metric.threshold == 0.5


class TestBusinessPolicyComplianceMetric:
    def test_pass(self) -> None:
        judge = StubJudge(score=10.0, reason="compliant")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 1.0
        assert metric.success is True
        assert metric.reason == "compliant"
        assert metric.error is None

    def test_fail(self) -> None:
        judge = StubJudge(score=0.0, reason="promised a refund")
        metric = BusinessPolicyComplianceMetric(judge)

        score = metric.measure(make_test_case())

        assert score == 0.0
        assert metric.success is False
        assert metric.reason == "promised a refund"

    def test_prompt_contains_policy(self) -> None:
        judge = StubJudge()
        metric = BusinessPolicyComplianceMetric(judge)

        metric.measure(make_test_case())

        prompt = judge.prompts[0]
        assert "Refund policy" in prompt
        assert "Privacy policy" in prompt
        assert "Guarantee policy" in prompt
        assert "Returns use a 30-day window." in prompt

    def test_name_and_threshold(self) -> None:
        metric = BusinessPolicyComplianceMetric(StubJudge())

        assert metric.name == "Business Policy Compliance"
        assert metric.threshold == 0.5


class TestEvaluateScenarioIntegration:
    def test_custom_metric_pass_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge(score=10.0))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 1.0
        assert scores[0].success is True
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_fail_recorded(self) -> None:
        metric = BusinessPolicyComplianceMetric(StubJudge(score=0.0))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Business Policy Compliance", metric)]
        )

        assert len(scores) == 1
        assert scores[0].score == 0.0
        assert scores[0].success is False
        assert scores[0].threshold == 0.5
        assert scores[0].error is None

    def test_custom_metric_error_recorded(self) -> None:
        metric = ToolOutcomeConsistencyMetric(StubJudge(error="boom"))

        scores = evaluate_scenario(
            make_scenario(), make_run(), [("Tool Outcome Consistency", metric)]
        )

        assert len(scores) == 1
        assert scores[0].error is not None
        assert "boom" in scores[0].error
        assert scores[0].success is None


class TestPolicyAwareTaskCompletion:
    def test_template_prompt_states_policy_completion_rules(self) -> None:
        prompt = PolicyAwareTaskCompletionTemplate.generate_verdict(
            task="Cancel order #1056",
            actual_outcome="Cannot cancel; the order has already shipped.",
        )

        assert "policy-compliant resolution" in prompt
        assert "correct refusal" in prompt
        assert "escalation" in prompt
        assert "order ID" in prompt
        assert "outside the 30-day return window" in prompt
        assert "violate store policy" in prompt
        assert "genuinely incomplete or incorrect" in prompt
        assert "Cancel order #1056" in prompt
        assert "Cannot cancel; the order has already shipped." in prompt

    def test_template_rewards_unavailable_information(self) -> None:
        prompt = PolicyAwareTaskCompletionTemplate.generate_verdict(
            task="Why is order #1045 delayed? I need it today.",
            actual_outcome="The order has shipped; no delivery date is available.",
        )

        assert "Reporting the actual order status" in prompt
        assert (
            "no delivery date, tracking information, or delay reason is currently "
            "available" in prompt
        )
        assert "without promising a callback or a response time" in prompt
        assert (
            "Do not penalize the agent for failing to explain a delay reason" in prompt
        )

    def test_metric_uses_policy_aware_template(self) -> None:
        metric = policy_aware_task_completion_metric(StubJudge())

        assert metric.evaluation_template is PolicyAwareTaskCompletionTemplate
        assert metric.__name__ == "Task Completion"
        assert metric.threshold == 0.5

    def test_metric_prompt_uses_the_override(self) -> None:
        metric = policy_aware_task_completion_metric(StubJudge())

        prompt = metric._get_prompt(
            "generate_verdict", task="Refund order #1064", actual_outcome="Denied."
        )

        assert "policy-compliant resolution" in prompt
        assert "Refund order #1064" in prompt
        assert "Denied." in prompt

    def test_build_metrics_uses_policy_aware_task_completion(self) -> None:
        metrics = dict(build_metrics(StubJudge()))

        task_completion = metrics["Task Completion"]
        assert task_completion.evaluation_template is PolicyAwareTaskCompletionTemplate
        assert task_completion.model is metrics["Tool Correctness"].model

    def test_task_is_the_customer_message_even_when_trace_extraction_is_empty(
        self,
    ) -> None:
        judge = TaskCompletionJudge()
        metric = policy_aware_task_completion_metric(judge)
        test_case = build_test_case(make_scenario(), make_run())
        test_case._trace_dict = {
            "name": "customer_support_agent",
            "type": "agent",
            "input": {"input": ""},
            "output": {"output": ""},
            "children": [],
        }

        metric.measure(test_case)

        assert judge.prompts
        assert make_scenario().customer_message in judge.prompts[0]

    def test_task_completion_judge_receives_tool_results(self) -> None:
        judge = TaskCompletionJudge()
        metric = policy_aware_task_completion_metric(judge)

        metric.measure(make_test_case())

        prompt = " ".join(judge.prompts[0].split())
        assert "Tool results and trace" in prompt
        assert "check_refund_eligibility" in prompt
        assert "eligible" in prompt
        assert "grounded evidence" in prompt


class TestPolicyAwareStepEfficiency:
    def test_template_prompt_states_calibration_rules(self) -> None:
        prompt = PolicyAwareStepEfficiencyTemplate.get_execution_efficiency(
            task="Cancel order #1055",
            trace_json_str='{"name": "customer_support_agent"}',
        )

        assert "Required pre-checks are efficient" in prompt
        assert "get_order_status before cancel_order" in prompt
        assert "Relevant additional information is efficient" in prompt
        assert "Still penalize genuine inefficiency" in prompt
        assert "repeated or duplicate calls" in prompt
        assert "Cancel order #1055" in prompt
        assert '{"name": "customer_support_agent"}' in prompt

    def test_template_treats_required_escalation_as_necessary(self) -> None:
        prompt = PolicyAwareStepEfficiencyTemplate.get_execution_efficiency(
            task="Refund order #1053 (cracked screen)",
            trace_json_str="TRACE",
        )

        assert "Required terminal actions are efficient" in prompt
        assert "escalate_to_human" in prompt
        assert "damaged, missing, or tampered item" in prompt
        assert "urgent delivery issue" in prompt
        assert "explicit human request" in prompt

    def test_template_requires_full_score_for_only_required_calls(self) -> None:
        prompt = PolicyAwareStepEfficiencyTemplate.get_execution_efficiency(
            task="t", trace_json_str="tr"
        )

        assert "A trace with only required calls scores 1.0" in prompt
        assert "the score must be 1.0" in prompt

    def test_template_treats_missing_identifier_request_as_efficient(self) -> None:
        prompt = PolicyAwareStepEfficiencyTemplate.get_execution_efficiency(
            task="I don't have my order number, but tell me what I bought last week.",
            trace_json_str="TRACE",
        )

        assert "Missing required information" in prompt
        assert (
            "asking the customer for it is the correct and efficient action" in prompt
        )
        assert "the absence of tool calls is efficient, not an" in prompt
        assert "Do not expect order tools to be called without the required" in prompt

    def test_metric_uses_policy_aware_template(self) -> None:
        metric = policy_aware_step_efficiency_metric(StubJudge())

        assert metric.evaluation_template is PolicyAwareStepEfficiencyTemplate
        assert metric.__name__ == "Step Efficiency"
        assert metric.threshold == 0.5

    def test_metric_prompt_uses_the_override(self) -> None:
        metric = policy_aware_step_efficiency_metric(StubJudge())

        prompt = metric._get_prompt(
            "get_execution_efficiency",
            task="What items are in order #1046?",
            trace_json_str="TRACE",
            multimodal=False,
        )

        assert "Required pre-checks are efficient" in prompt
        assert "What items are in order #1046?" in prompt
        assert "TRACE" in prompt

    def test_build_metrics_uses_policy_aware_step_efficiency(self) -> None:
        metrics = dict(build_metrics(StubJudge()))

        step_efficiency = metrics["Step Efficiency"]
        assert step_efficiency.evaluation_template is PolicyAwareStepEfficiencyTemplate
        assert step_efficiency.model is metrics["Tool Correctness"].model

    def test_final_score_equals_the_judge_verdict_score(self) -> None:
        judge = StepEfficiencyJudge(score=1.0, reason="All actions were necessary.")
        metric = policy_aware_step_efficiency_metric(judge)

        metric.measure(build_test_case(make_scenario(), make_run()))

        assert metric.score == 1.0
        assert metric.success is True
        assert metric.reason == "All actions were necessary."

    def test_verdict_score_is_passed_through_without_conversion(self) -> None:
        judge = StepEfficiencyJudge(score=0.1, reason="described as efficient")
        metric = policy_aware_step_efficiency_metric(judge)

        metric.measure(build_test_case(make_scenario(), make_run()))

        assert metric.score == 0.1
        assert metric.success is False
        assert metric.reason == "described as efficient"


def make_cancellation_scenario(
    scenario_id: str, message: str, order_id: str, expected_tool_calls: list[str]
) -> Scenario:
    return Scenario(
        id=scenario_id,
        category="cancellation",
        customer_message=message,
        order_id=order_id,
        expected_tool_calls=expected_tool_calls,
    )


class TestPolicyAwareArgumentCorrectness:
    def test_template_includes_cancellation_policy(self) -> None:
        prompt = PolicyAwareArgumentCorrectnessTemplate.generate_verdicts(
            input="Cancel order #1056.",
            stringified_tools_called="[ToolCall(name='get_order_status')]",
        )

        assert (
            "A cancellation is allowed only for orders whose status is 'processing'."
            in prompt
        )
        assert "verify its status with get_order_status" in prompt
        assert (
            "do not mark it 'no' merely because the agent did not call cancel_order"
            in prompt
        )
        assert "cancel_order for a shipped order" in prompt
        assert "Cancel order #1056." in prompt
        assert "[ToolCall(name='get_order_status')]" in prompt

    def test_metric_uses_policy_aware_template(self) -> None:
        metric = policy_aware_argument_correctness_metric(StubJudge())

        assert metric.evaluation_template is PolicyAwareArgumentCorrectnessTemplate
        assert metric.__name__ == "Argument Correctness"
        assert metric.threshold == 0.5

    def test_build_metrics_uses_policy_aware_argument_correctness(self) -> None:
        metrics = dict(build_metrics(StubJudge()))

        argument_correctness = metrics["Argument Correctness"]
        assert (
            argument_correctness.evaluation_template
            is PolicyAwareArgumentCorrectnessTemplate
        )
        assert argument_correctness.model is metrics["Tool Correctness"].model

    def test_status_lookup_for_shipped_order_is_not_penalized(self) -> None:
        judge = ArgumentCorrectnessJudge(verdict=Verdict.YES)
        metric = policy_aware_argument_correctness_metric(judge)
        scenario = make_cancellation_scenario(
            "cancellation_002",
            "Cancel order #1056. It has already shipped, but I changed my mind.",
            "1056",
            ["get_order_status"],
        )
        run = AgentRun(
            run_id="r",
            user_message=scenario.customer_message,
            model="gemini",
            answer="Order #1056 has already shipped, so it cannot be cancelled.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="get_order_status",
                    arguments={"order_id": "1056"},
                    result='{"status": "shipped"}',
                )
            ],
        )

        metric.measure(build_test_case(scenario, run))

        assert metric.score == 1.0
        assert metric.success is True

    def test_valid_cancellation_arguments_pass(self) -> None:
        judge = ArgumentCorrectnessJudge(verdict=Verdict.YES)
        metric = policy_aware_argument_correctness_metric(judge)
        scenario = make_cancellation_scenario(
            "cancellation_001",
            "Please cancel order #1055. I ordered the wrong cable.",
            "1055",
            ["get_order_status", "cancel_order"],
        )
        run = AgentRun(
            run_id="r",
            user_message=scenario.customer_message,
            model="gemini",
            answer="Order #1055 has been successfully cancelled.",
            tool_calls=[
                ToolCallRecord(
                    tool_name="get_order_status",
                    arguments={"order_id": "1055"},
                    result='{"status": "processing"}',
                ),
                ToolCallRecord(
                    tool_name="cancel_order",
                    arguments={"order_id": "1055"},
                    result='{"cancelled": true}',
                ),
            ],
        )

        metric.measure(build_test_case(scenario, run))

        assert metric.score == 1.0
        assert metric.success is True

    def test_incorrect_arguments_fail(self) -> None:
        judge = ArgumentCorrectnessJudge(verdict=Verdict.NO, reason="wrong order id")
        metric = policy_aware_argument_correctness_metric(judge)

        metric.measure(build_test_case(make_scenario(), make_run()))

        assert metric.score == 0.0
        assert metric.success is False
