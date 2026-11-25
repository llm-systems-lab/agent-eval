from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from pydantic import Field, PrivateAttr

from agent.baseline_agent import BaselineAgent, create_baseline_agent
from agent.config import AgentSettings
from agent.tools import DEFAULT_DATA_PATH, EscalationRegistry, OrderStore

RAW_STORE = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))


def make_store() -> OrderStore:
    return OrderStore(data=RAW_STORE)


def make_agent(
    model: BaseChatModel,
    *,
    recursion_limit: int = 25,
) -> BaselineAgent:
    settings = AgentSettings(google_api_key="dummy-key", model="fake-model")
    return create_baseline_agent(
        model=model,
        settings=settings,
        store=make_store(),
        registry=EscalationRegistry(),
        configure_tracing=False,
        recursion_limit=recursion_limit,
    )


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted assistant messages, one per model turn."""

    responses: list[AIMessage] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Any, **kwargs: Any
    ) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._index >= len(self.responses):
            raise AssertionError("scripted model has no further responses")
        message = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class RepeatingToolCallChatModel(BaseChatModel):
    """Always emits the same tool call so the agent loops until the limit."""

    tool_name: str = "get_order_status"
    tool_args: dict[str, Any] = Field(default_factory=lambda: {"order_id": "1042"})

    @property
    def _llm_type(self) -> str:
        return "repeating-tool-call"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RepeatingToolCallChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": self.tool_args,
                    "id": f"call-{uuid.uuid4().hex[:8]}",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class ErrorChatModel(BaseChatModel):
    """Raises on every model call to simulate an API failure."""

    @property
    def _llm_type(self) -> str:
        return "error"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ErrorChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("simulated API failure")


class TestBaselineAgentOrchestration:
    def test_response_without_tool_calls(self) -> None:
        model = ScriptedChatModel(
            responses=[AIMessage(content="Hello, how can I help you today?")]
        )
        agent = make_agent(model)

        run = agent.run("Hi there")

        assert run.tool_calls == []
        assert run.answer == "Hello, how can I help you today?"

    def test_single_tool_call_is_captured_with_result(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1042"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="Order 1042 is processing."),
            ]
        )
        agent = make_agent(model)

        run = agent.run("What is the status of order 1042?")

        assert len(run.tool_calls) == 1
        record = run.tool_calls[0]
        assert record.tool_name == "get_order_status"
        assert record.arguments == {"order_id": "1042"}
        assert record.error is None
        result = json.loads(record.result or "{}")
        assert result["success"] is True
        assert result["status"] == "processing"
        assert run.answer == "Order 1042 is processing."

    def test_multiple_tool_calls_across_steps_are_ordered(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1055"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "cancel_order",
                            "args": {"order_id": "1055"},
                            "id": "call-2",
                        }
                    ],
                ),
                AIMessage(content="Order 1055 has been cancelled."),
            ]
        )
        agent = make_agent(model)

        run = agent.run("Please cancel order 1055.")

        assert [record.tool_name for record in run.tool_calls] == [
            "get_order_status",
            "cancel_order",
        ]
        cancel_result = json.loads(run.tool_calls[1].result or "{}")
        assert cancel_result["cancelled"] is True
        assert run.answer == "Order 1055 has been cancelled."

    def test_parallel_tool_calls_in_one_message(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1046"},
                            "id": "call-a",
                        },
                        {
                            "name": "get_order_details",
                            "args": {"order_id": "1046"},
                            "id": "call-b",
                        },
                    ],
                ),
                AIMessage(content="Here are the status and details."),
            ]
        )
        agent = make_agent(model)

        run = agent.run("Tell me about order 1046.")

        assert [record.tool_name for record in run.tool_calls] == [
            "get_order_status",
            "get_order_details",
        ]
        assert all(record.result is not None for record in run.tool_calls)

    def test_tool_calls_are_read_from_messages_not_answer(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1042"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="The order status is confirmed."),
            ]
        )
        agent = make_agent(model)

        run = agent.run("Status please")

        assert run.answer == "The order status is confirmed."
        assert [record.tool_name for record in run.tool_calls] == ["get_order_status"]

    def test_stream_emits_tool_call_then_result_then_final(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1042"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="Order 1042 is processing."),
            ]
        )
        agent = make_agent(model)

        events = list(agent.stream("Status of 1042?"))

        kinds = [event.kind for event in events]
        assert kinds == ["tool_call", "tool_result", "final"]
        assert events[0].tool_call is not None
        assert events[0].tool_call.tool_name == "get_order_status"
        assert events[1].tool_call is not None
        assert events[1].tool_call.result is not None
        assert events[-1].run is not None
        assert events[-1].run.answer == "Order 1042 is processing."

    def test_api_error_propagates(self) -> None:
        agent = make_agent(ErrorChatModel())

        with pytest.raises(RuntimeError, match="simulated API failure"):
            agent.run("anything")

    def test_recursion_limit_stops_infinite_tool_loop(self) -> None:
        agent = make_agent(RepeatingToolCallChatModel(), recursion_limit=4)

        with pytest.raises(GraphRecursionError):
            agent.run("keep calling the tool")
