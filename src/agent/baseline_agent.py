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
    serialize_messages,
)

BASELINE_TAG = "baseline"
BASELINE_RUN_NAME = "baseline-agent"
BASELINE_METADATA: dict[str, Any] = {
    "experiment": "baseline",
    "agent_version": 1,
}
DEFAULT_RECURSION_LIMIT = 25

SYSTEM_PROMPT = """You are a customer support assistant for a fictional electronics store.

Help customers with order status, order details, refunds, and cancellations.
Use the available tools when you need order information.
Do not invent order information.
Communicate clearly and politely.
"""


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
        self.metadata = dict(BASELINE_METADATA if metadata is None else metadata)
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
        """Invoke the agent once and return a structured execution trace."""
        started_at = datetime.now(UTC)
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=self.run_config,
        )
        finished_at = datetime.now(UTC)
        messages: list[BaseMessage] = result["messages"]
        return self._build_run(user_message, messages, started_at, finished_at)

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
                            arguments=dict(call.get("args") or {}),
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
        temperature=settings.temperature,
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
