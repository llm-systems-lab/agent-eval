"""Custom DeepEval metrics for semantic behavior checks.

The two GEval metrics reuse the existing OpenRouter judge
(``src/agent/judge.py``) to evaluate the final answer against the tool results
and the store policy without duplicating the deterministic checks in
``behavior_checks``. ``PolicyAwareTaskCompletionTemplate`` customizes DeepEval's
Task Completion prompt so a policy-compliant resolution counts as completion,
and ``PolicyAwareStepEfficiencyTemplate`` calibrates Step Efficiency so required
pre-checks and helpful details are not penalized.

The scenario and agent run are read from ``LLMTestCase.metadata``, which
``metric_evaluation.build_test_case`` fills in. A judge or evaluation failure is
recorded by the runner as a metric error; it is never treated as a pass.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.metrics import GEval
from deepeval.metrics.argument_correctness import (
    ArgumentCorrectnessMetric,
    ArgumentCorrectnessTemplate,
)
from deepeval.metrics.step_efficiency import (
    StepEfficiencyMetric,
    StepEfficiencyTemplate,
)
from deepeval.metrics.task_completion import (
    TaskCompletionMetric,
    TaskCompletionTemplate,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

CANCELLATION_POLICY = """Cancellation policy:
- Cancel only orders whose status is processing.
- Report the actual cancellation result from the tool; never claim a cancellation
  succeeded unless the tool result confirms it."""

STORE_POLICY = f"""Refund policy:
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

{CANCELLATION_POLICY}

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
        "Saying that a specific detail is not available (for example, no delivery "
        "date, tracking information, or delay reason) is supported when the "
        "recorded tool results do not contain that detail: do not treat a truthful "
        "'not available' as unsupported. Do not accept a claim that a tool "
        "explicitly provided or confirmed information it did not, and do not accept "
        "invented details."
    ),
    (
        "Treat a claim as unsupported when the tool results contradict it, are "
        "missing, or do not establish the specific outcome."
    ),
    (
        "A policy claim may be supported by the store's business policy instead of "
        "a tool result. The relevant business policy is:\n" + CANCELLATION_POLICY
    ),
    (
        "Treat a claim as supported when it matches the business policy above, even "
        "without a tool call: for example, a statement that a shipped order cannot "
        "be cancelled is supported by the cancellation policy. Do not treat an "
        "arbitrary claim as policy-backed when the policy above does not support it, "
        "and still require tool evidence for claims about completed actions, "
        "statuses, refunds, or successful escalations."
    ),
    (
        "A claim is also supported when the customer themselves stated that fact in "
        "the input: for example, 'the laptop arrived with a cracked screen' is "
        "supported because the customer reported it, and does not require tool "
        "verification. Do not treat the customer's own description of their issue as "
        "an unsupported claim."
    ),
    (
        "Score 10 when every claim in the final answer is supported by the tool "
        "results or the business policy above, and 0 when any claim is unsupported. "
        "In the reason, name the unsupported claim and its source."
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
            async_mode=False,
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
            async_mode=False,
            _include_g_eval_suffix=False,
        )


class PolicyAwareTaskCompletionTemplate(TaskCompletionTemplate):  # type: ignore[misc]
    """Task Completion prompt that treats policy-compliant resolutions as complete."""

    @staticmethod
    def generate_verdict(
        task: str, actual_outcome: str, tool_results: str = ""
    ) -> str:
        return (
            "Given the task (desired outcome) and the actual achieved outcome, "
            "compare how well the actual outcome aligns with the desired task.\n\n"
            "The agent is a customer support agent bound by store policy. Judge "
            "task completion by whether the agent correctly resolved the "
            "customer's request under that policy, not by whether it did exactly "
            "what the customer literally asked for.\n\n"
            "Count the task as completed (high verdict) when the actual outcome is "
            "a policy-compliant resolution, including:\n"
            "- A correct refusal to refund, cancel, or act because store policy "
            "forbids it, with a clear explanation of why it is unavailable.\n"
            "- A correct escalation to human support when escalation is required "
            "(for example damaged, missing, tampered, or unverifiable items, an "
            "unresolvable request, or an explicit human request).\n"
            "- Asking the customer for an order ID when it is required to look up "
            "or protect the order, including for privacy.\n"
            "- Explaining that an order cannot be cancelled because it has already "
            "shipped, or that a refund is outside the 30-day return window.\n"
            "- Reporting the actual order status returned by the tool (for example "
            "that the order has shipped).\n"
            "- Explaining that no delivery date, tracking information, or delay "
            "reason is currently available when the tool results provide none.\n"
            "- Providing an appropriate next step, such as escalating an urgent "
            "issue, without promising a callback or a response time.\n\n"
            "Do not lower the verdict for declining an action that would violate "
            "store policy, and never require the agent to take such an action.\n"
            "Do not penalize the agent for failing to explain a delay reason, "
            "delivery date, or tracking information that the tool results do not "
            "provide: the agent cannot invent it, so reward it for reporting what "
            "the tool returned and stating that the rest is unavailable.\n\n"
            "Still give a low verdict when the actual outcome is genuinely "
            "incomplete or incorrect: it ignores the request, gives wrong or "
            "unfounded information, fails to explain a refusal, or omits a "
            "required action that policy allows.\n\n"
            "Verify the actual outcome against the recorded tool results and trace "
            "below: treat details that match the tool results as grounded evidence, "
            "not fabrication, and base the verdict on what the tools actually "
            "returned rather than assuming the outcome was invented.\n\n"
            "Return a JSON with two keys: `verdict` and `reason`.\n"
            "- The `verdict` should be a score from 0 to 1, where 1 indicates the "
            "actual outcome fully and correctly resolves the task under store "
            "policy, and 0 indicates it does not.\n"
            "- The `reason` should explain why the given verdict was assigned.\n\n"
            "**\n"
            "IMPORTANT: Please make sure to only return in JSON format, with "
            "`verdict` as a float between 0 and 1.\n"
            "**\n\n"
            f"Task:\n{task}\n\n"
            f"Actual outcome:\n{actual_outcome}\n\n"
            f"Tool results and trace:\n{tool_results}\n\n"
            "JSON:"
        )


class PolicyAwareTaskCompletionMetric(TaskCompletionMetric):  # type: ignore[no-untyped-call]
    """Task Completion that uses the customer message as the task.

    DeepEval derives the task by LLM-extracting it from the execution trace,
    which can return an empty task. This metric uses ``test_case.input`` (the
    scenario's customer message) instead, so the task is always deterministic.
    """

    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        super().__init__(
            model=model,
            threshold=threshold,
            async_mode=False,
            evaluation_template=PolicyAwareTaskCompletionTemplate,
        )
        self._tool_context = ""

    @staticmethod
    def _format_tool_context(test_case: LLMTestCase) -> str:
        """Render the recorded tool results/trace so the judge can verify the answer."""
        trace = getattr(test_case, "_trace_dict", None)
        if isinstance(trace, dict):
            return json.dumps(trace, indent=2, default=str)
        tools_called = getattr(test_case, "tools_called", None) or []
        return json.dumps(
            [
                {
                    "name": getattr(call, "name", None),
                    "input_parameters": getattr(call, "input_parameters", None),
                    "output": getattr(call, "output", None),
                }
                for call in tools_called
            ],
            indent=2,
            default=str,
        )

    def _extract_task_and_outcome(self, test_case: LLMTestCase) -> tuple[str, Any]:
        self._tool_context = self._format_tool_context(test_case)
        return test_case.input, test_case.actual_output

    async def _a_extract_task_and_outcome(
        self, test_case: LLMTestCase
    ) -> tuple[str, Any]:
        self._tool_context = self._format_tool_context(test_case)
        return test_case.input, test_case.actual_output

    def _get_prompt(
        self, method: Any, *, template_class: Any = None, **kwargs: Any
    ) -> str:
        # DeepEval resolves internal prompts (such as the experimental
        # system-one score) by the metric's class name. Pin the base name so
        # this differently-named subclass still resolves them.
        if method == "generate_verdict" and "tool_results" not in kwargs:
            kwargs["tool_results"] = self._tool_context
        if template_class is None and getattr(
            self.evaluation_template, method, None
        ) is None:
            template_class = "TaskCompletionMetric"
        return super()._get_prompt(method, template_class=template_class, **kwargs)


def policy_aware_task_completion_metric(
    model: Any, threshold: float = 0.5
) -> TaskCompletionMetric:
    """Build a Task Completion metric that counts policy-compliant resolutions."""
    return PolicyAwareTaskCompletionMetric(model=model, threshold=threshold)


class PolicyAwareStepEfficiencyTemplate(StepEfficiencyTemplate):  # type: ignore[misc]
    """Step Efficiency prompt that does not penalize required steps or helpful details."""

    @staticmethod
    def get_execution_efficiency(task: str, trace_json_str: str) -> str:
        return (
            "You are an efficiency auditor evaluating how economically an AI "
            "customer support agent executed a task.\n\n"
            "OBJECTIVE:\n\n"
            "Determine how efficiently the agent executed the given task based on "
            "its full execution trace. Efficiency means achieving the customer's "
            "goal using the fewest, simplest, and most direct actions that store "
            "policy allows.\n\n"
            "Assign a score from 0.0 to 1.0 that reflects how close the execution "
            "came to the minimal necessary sequence of actions.\n\n"
            "You are not evaluating correctness, completeness, or helpfulness — "
            "only the efficiency of the execution.\n\n"
            "CALIBRATION RULES:\n\n"
            "1. Required pre-checks are efficient, not redundant.\n"
            "- A tool call that store policy or the tool contract requires before "
            "another action is necessary. For example, checking order status with "
            "get_order_status before cancel_order is required to confirm the order "
            "is cancellable, so it must not be penalized.\n"
            "- Verifying eligibility, ownership, or state before acting is "
            "efficient when the action depends on it.\n\n"
            "2. Required terminal actions are efficient, not redundant.\n"
            "- An action that store policy requires to resolve the request is "
            "necessary even when it is the final step and nothing follows it. "
            "Being the last action does not make it an extra step.\n"
            "- Calling escalate_to_human is necessary, not an extra step, when the "
            "request involves a damaged, missing, or tampered item, an urgent "
            "delivery issue, or an explicit human request.\n"
            "- Do not penalize a required escalation, and do not expect the agent "
            "to take an action that store policy forbids (for example cancel_order "
            "for a shipped order, or processing a refund).\n\n"
            "3. Relevant additional information is efficient.\n"
            "- Including accurate, relevant details in the final answer that help "
            "the customer (for example the order date, delivery date, or total when "
            "listing items) is not an inefficiency when the details come from tool "
            "results already retrieved.\n"
            "- Do not penalize an answer for being helpful or complete.\n\n"
            "4. Still penalize genuine inefficiency.\n"
            "- Penalize tool calls that are not required by the task or store "
            "policy, speculative or exploratory calls, repeated or duplicate calls "
            "for the same data, calls whose result is ignored, and actions "
            "unrelated to the customer's request.\n"
            "- Penalize redundant reasoning or retrieval loops.\n\n"
            "5. Minimal action principle.\n"
            "- The ideal execution performs the minimum number of steps needed to "
            "resolve the request under store policy, including any required "
            "pre-checks and required terminal actions.\n\n"
            "6. A trace with only required calls scores 1.0.\n"
            "- If every tool call in the trace is required by the task or store "
            "policy (including required pre-checks and required terminal actions) "
            "and there are no speculative, redundant, or unrelated actions, assign "
            "a score of 1.0.\n\n"
            "7. When in doubt.\n"
            "- If it is genuinely unclear whether an action was required, do not "
            "assume it was unnecessary when store policy or the tool contract "
            "plausibly requires it.\n\n"
            "8. Missing required information.\n"
            "- When the request cannot be fulfilled without an order ID or another "
            "identifier the customer has not provided, asking the customer for it is "
            "the correct and efficient action: it is a required step before any "
            "lookup. In that situation the absence of tool calls is efficient, not an "
            "inefficiency, and must not be penalized.\n"
            "- Do not expect order tools to be called without the required "
            "identifier.\n\n"
            "SCORING SCALE\n\n"
            "Choose the score only from these values, and make the reason consistent "
            "with the score:\n"
            "- 1.0 — Perfectly efficient: only necessary steps, including required "
            "pre-checks and required terminal actions; no speculative, redundant, or "
            "unrelated actions.\n"
            "- 0.75 — Strong efficiency: mostly minimal with one small redundant "
            "step.\n"
            "- 0.5 — Moderate efficiency: noticeable inefficiency such as an "
            "unnecessary tool call or an indirect method.\n"
            "- 0.25 — Low efficiency: multiple irrelevant or unjustified actions.\n"
            "- 0.0 — Highly inefficient: most actions unnecessary or unrelated to "
            "achieving the core task.\n\n"
            "OUTPUT FORMAT:\n\n"
            "Return a single JSON object in this exact format:\n\n"
            "{\n"
            '  "score": 0.0,\n'
            '  "reason": "1-3 concise factual sentences describing where '
            'inefficiencies occurred."\n'
            "}\n\n"
            "The reason must identify specific inefficient actions and avoid "
            "subjective phrasing. If the reason describes only necessary actions, "
            "the score must be 1.0.\n\n"
            f"TASK:\n{task}\n\n"
            f"TRACE:\n{trace_json_str}\n\n"
            "JSON:"
        )


def policy_aware_step_efficiency_metric(
    model: Any, threshold: float = 0.5
) -> StepEfficiencyMetric:
    """Build a Step Efficiency metric calibrated for the customer support agent.

    DeepEval resolves unmodified prompts by class name, so the metric stays a
    plain ``StepEfficiencyMetric`` and only the efficiency prompt is overridden.
    """
    return StepEfficiencyMetric(
        model=model,
        threshold=threshold,
        async_mode=False,
        evaluation_template=PolicyAwareStepEfficiencyTemplate,
    )


class PolicyAwareArgumentCorrectnessTemplate(ArgumentCorrectnessTemplate):  # type: ignore[misc]
    """Argument Correctness prompt that applies store cancellation policy."""

    @staticmethod
    def generate_verdicts(input: str, stringified_tools_called: Any) -> str:
        return (
            "For the provided list of tool calls, determine whether each tool call "
            "input parameter is relevantly and correctly addresses the input.\n\n"
            "This agent is a customer support agent bound by store policy. Apply "
            "these rules when judging:\n"
            "- A cancellation is allowed only for orders whose status is "
            "'processing'. An order that has already shipped cannot be cancelled "
            "with the available tools.\n"
            "- Before cancelling an order, the agent must verify its status with "
            "get_order_status.\n"
            "- A status lookup (for example get_order_status called with the "
            "customer's order ID) is a correct tool call when the order cannot be "
            "cancelled: do not mark it 'no' merely because the agent did not call "
            "cancel_order.\n"
            "- Do not require the agent to call a tool that store policy forbids "
            "for the order (for example cancel_order for a shipped order).\n\n"
            "Please generate a list of JSON with two keys: `verdict` and `reason`.\n"
            "The 'verdict' key should STRICTLY be either a 'yes' or 'no'. Answer "
            "'yes' if the tool call input parameter relevantly and correctly "
            "addresses the original input and the tool call is permitted and "
            "appropriate under store policy. Answer 'no' if the parameter does not "
            "correctly address the original input, or the tool call violates store "
            "policy.\n"
            "The 'reason' is the reason for the verdict.\n"
            "Provide a 'reason' ONLY if the answer is 'no'.\n"
            "If there is no input parameter, answer 'no' for the verdict and "
            "provide the reason as 'No input parameter provided'.\n\n"
            "**\n"
            "IMPORTANT: Please make sure to only return in valid and parseable JSON "
            "format, with the 'verdicts' key mapping to a list of JSON objects. "
            "Ensure all strings are closed appropriately. Repair any invalid JSON "
            "before you output it.\n"
            "Example input:\n"
            '"What items did I buy in order #1046?"\n'
            "Example tool calls:\n"
            "[ToolCall(name='get_order_details', input_parameters={'order_id': "
            "'1046'}, output={'items': ['Wireless Mouse']})]\n"
            "Example JSON:\n"
            "{\n"
            '  "verdicts": [\n'
            '    {\n'
            '      "verdict": "yes"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "===== END OF EXAMPLE ======\n\n"
            "Since you are going to generate a verdict for each statement, the "
            "number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of "
            "`statements`.\n"
            "**\n\n"
            f"Input:\n{input}\n\n"
            f"Tool Calls:\n{stringified_tools_called}\n\n"
            "JSON:"
        )


def policy_aware_argument_correctness_metric(
    model: Any, threshold: float = 0.5
) -> ArgumentCorrectnessMetric:
    """Build an Argument Correctness metric calibrated for the customer support agent.

    DeepEval resolves unmodified prompts by class name, so the metric stays a
    plain ``ArgumentCorrectnessMetric`` and only the verdict prompt is overridden.
    """
    return ArgumentCorrectnessMetric(
        model=model,
        threshold=threshold,
        async_mode=False,
        evaluation_template=PolicyAwareArgumentCorrectnessTemplate,
    )
