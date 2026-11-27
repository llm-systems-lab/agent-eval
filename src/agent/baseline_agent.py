"""Baseline customer support agent built with LangChain's ``create_agent``.

The agent uses the existing deterministic tools through thin wrappers and a
Google Gemini model. It holds no evaluation logic and no scenario-specific
branching; scenario content never reaches the agent.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    AgentSettings,
    load_agent_settings,
    load_langsmith_settings,
)
from .toolkit import build_customer_support_tools
from .tools.customer_support_tools import EscalationRegistry
from .tools.store import OrderStore
from .traces import (
    AgentEvent,
    AgentRun,
    ToolCallRecord,
    content_to_text,
    extract_answer,
    extract_tool_calls,
    normalize_tool_arguments,
    serialize_messages,
)

BASELINE_TAG = "baseline"
BASELINE_RUN_NAME = "baseline-agent"
BASELINE_METADATA: dict[str, Any] = {
    "experiment": "baseline",
    "agent_version": 1,
}
DEFAULT_RECURSION_LIMIT = 25
MAX_ANSWER_ATTEMPTS = 2

SYSTEM_PROMPT = """You are a customer support assistant for a fictional electronics store.

Help customers with order status, order details, refunds, and cancellations.
Use the available tools when you need order information. To check whether an order exists or its
current status, call get_order_status; use get_order_details only when the customer asks about the
order's items, totals, or dates.
Do not invent order information, response times, or follow-up details.
Communicate clearly and politely.

Refund policy:
Check eligibility with the check_refund_eligibility tool and explicitly state that you performed an
eligibility check before reporting the result, then explain whether the order qualifies for a return.
Report return eligibility only: say that the order is eligible for a return, and do not state that it
is eligible for, approved for, or guaranteed a refund.
Explain that checking refund eligibility does not itself process a refund. Do not claim whether a
refund has or has not previously been processed unless an actual tool result explicitly confirms it.
Do not claim that a refund has been processed or offer to process it unless a dedicated
refund-processing tool is available.

Cancellation policy:
To cancel an order, first check its current status with get_order_status. Only processing orders can
be cancelled. If the order has shipped, been delivered, or is already cancelled, do not call
cancel_order; instead explain that it cannot be cancelled through the self-service cancellation tool,
which can only cancel orders that are still processing, and base that explanation on the status
returned by get_order_status.

Delivery information:
When a customer asks about a delivery date, or why an order is delayed, and no tool result provides a
delivery date, you must explicitly tell the customer that no delivery date is currently available.
For example, if an order is still processing or shipped, state that no delivery date is currently
available; do not guess one. Never invent, estimate, or guarantee a delivery date.

Escalation policy:
Call the escalate_to_human tool immediately when any of these conditions apply, and include a short
reason. Do not merely offer to escalate or ask the customer for permission - perform the escalation
yourself.
If the customer has not provided an order ID, ask for it before looking up order-specific details; do
not escalate merely because the ID is missing. Escalate the not-found case only after checking the ID
the customer provided.
- The order cannot be found in the system after you check the order ID the customer provided.
- The customer reports a delivery delay and explicitly states urgency or an unmet delivery deadline.
- An item is damaged, missing, tampered with, or cannot be verified.
- The customer explicitly asks to speak to a human representative.
- The complaint remains unresolved after you have already tried to help.

Delivery or package disputes:
When the customer reports that a package was marked delivered but not received, or another
delivery-tracking problem (not a damaged item), call get_order_status first (not get_order_details),
state what the status shows, and then escalate the dispute. Do not decide the claim outcome yourself.

Escalation responses:
When you escalate because of a damaged, missing, or tampered item, explicitly acknowledge the
customer's specific claim in your reply (for example, that the item arrived damaged) rather than
replying with only a generic escalation confirmation. If the customer asks for a refund or return for
such an item, state that human support will review the refund request, and do not promise or guarantee
the refund.
If the customer is upset or dissatisfied, acknowledge their frustration empathetically before
confirming the escalation.

Follow-up and callback policy:
Never promise a callback, a response time, or any follow-up timeframe unless a tool result explicitly
confirms it. Do not invent what happens next or when someone will respond.
When escalate_to_human succeeds, you may confirm that the case was escalated and give the reference
number, using neutral wording, for example:
"Your case has been escalated to our support team. Your reference number is ESC-000001."
Do not add a promise about when or how the support team will follow up.
"""


def _is_substantive_answer(text: str) -> bool:
    """Return True when a final answer looks like a real customer reply.

    Empty content, or a stray non-Latin fragment, indicates a degenerate model
    completion; the caller retries rather than surfacing it verbatim.
    """
    return any(char.isascii() and char.isalnum() for char in text)


class BaselineAgent:
    """Thin wrapper that runs the compiled agent graph and records a trace."""

    def __init__(
        self,
        graph: Any,
        *,
        model_name: str,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        tags: Sequence[str] = (BASELINE_TAG,),
        metadata: Mapping[str, Any] | None = None,
        run_name: str = BASELINE_RUN_NAME,
    ) -> None:
        self._graph = graph
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self.tags = list(tags)
        self.metadata = {**BASELINE_METADATA, **(metadata or {})}
        self.run_name = run_name

    @property
    def run_config(self) -> RunnableConfig:
        """Run configuration used for tracing and the recursion limit."""
        return {
            "recursion_limit": self.recursion_limit,
            "tags": self.tags,
            "metadata": self.metadata,
            "run_name": self.run_name,
        }

    def _build_run(
        self,
        user_message: str,
        messages: Sequence[BaseMessage],
        started_at: datetime,
        finished_at: datetime,
    ) -> AgentRun:
        return AgentRun(
            run_id=uuid.uuid4().hex,
            user_message=user_message,
            model=self.model_name,
            answer=extract_answer(messages),
            tool_calls=extract_tool_calls(messages),
            messages=serialize_messages(messages),
            started_at=started_at,
            finished_at=finished_at,
        )

    def run(self, user_message: str) -> AgentRun:
        """Invoke the agent and return a structured execution trace.

        A degenerate final answer (empty, or a stray non-Latin fragment) is a
        model glitch rather than a real reply; such a completion is retried up
        to ``MAX_ANSWER_ATTEMPTS`` times, and the last attempt is returned.
        """
        run: AgentRun | None = None
        for _ in range(MAX_ANSWER_ATTEMPTS):
            started_at = datetime.now(UTC)
            result = self._graph.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=self.run_config,
            )
            finished_at = datetime.now(UTC)
            messages: list[BaseMessage] = result["messages"]
            run = self._build_run(user_message, messages, started_at, finished_at)
            if _is_substantive_answer(run.answer):
                break
        assert run is not None
        return run

    def stream(self, user_message: str) -> Iterator[AgentEvent]:
        """Stream an agent run, yielding tool-call and result events.

        The final event (``kind == "final"``) carries the same ``AgentRun`` that
        ``run`` would return, so CLI presentation and programmatic use share one
        implementation and one result contract.
        """
        started_at = datetime.now(UTC)
        seen = 0
        records: dict[str, ToolCallRecord] = {}
        messages: list[BaseMessage] = []
        for chunk in self._graph.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=self.run_config,
            stream_mode="values",
        ):
            messages = chunk["messages"]
            for message in messages[seen:]:
                if isinstance(message, AIMessage) and message.tool_calls:
                    for call in message.tool_calls:
                        record = ToolCallRecord(
                            tool_name=call.get("name", ""),
                            call_id=call.get("id"),
                            arguments=normalize_tool_arguments(
                                dict(call.get("args") or {})
                            ),
                        )
                        if record.call_id is not None:
                            records[record.call_id] = record
                        yield AgentEvent(kind="tool_call", tool_call=record)
                elif isinstance(message, ToolMessage):
                    matched = records.get(message.tool_call_id or "")
                    if matched is not None:
                        text = content_to_text(message.content)
                        if getattr(message, "status", "success") == "error":
                            matched.error = text
                        else:
                            matched.result = text
                        yield AgentEvent(kind="tool_result", tool_call=matched)
            seen = len(messages)

        finished_at = datetime.now(UTC)
        yield AgentEvent(
            kind="final",
            run=self._build_run(user_message, messages, started_at, finished_at),
        )


def _build_model(settings: AgentSettings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=settings.model,
        api_key=settings.google_api_key,
    )


def create_baseline_agent(
    *,
    store: OrderStore | None = None,
    registry: EscalationRegistry | None = None,
    reference_date: date | None = None,
    settings: AgentSettings | None = None,
    model: BaseChatModel | None = None,
    configure_tracing: bool = True,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    tags: Sequence[str] = (BASELINE_TAG,),
    metadata: Mapping[str, Any] | None = None,
    run_name: str = BASELINE_RUN_NAME,
) -> BaselineAgent:
    """Build the Baseline Agent.

    ``store`` and ``registry`` should be supplied with isolated instances when
    running scenarios concurrently, because cancellation and escalation mutate
    in-memory state. ``reference_date`` fixes the refund window for comparable
    runs. ``model`` and ``settings`` allow injecting a fake model in tests.

    ``tags``, ``metadata``, and ``run_name`` are passed to LangSmith so the
    Baseline and future Improved agents share one project and are distinguished
    by tags and metadata rather than separate projects.
    """
    if configure_tracing:
        load_langsmith_settings().apply()

    agent_settings = settings or load_agent_settings()
    active_model = model or _build_model(agent_settings)
    tools = build_customer_support_tools(
        store=store,
        registry=registry,
        reference_date=reference_date,
    )
    graph = create_agent(
        model=active_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return BaselineAgent(
        graph,
        model_name=agent_settings.model,
        recursion_limit=recursion_limit,
        tags=tags,
        metadata=metadata,
        run_name=run_name,
    )
