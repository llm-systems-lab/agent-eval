"""Structured execution traces for agent runs.

Traces capture the ordered tool calls, their arguments, and their results as
structured data rather than free-text logs. They are the basis for the later
behavior evaluation and failure investigation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field

from .tools.customer_support_tools import _normalize_order_id


class ToolCallRecord(BaseModel):
    """A single executed tool call and the result the agent received for it."""

    tool_name: str
    call_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None


def normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return tool-call arguments with ``order_id`` normalized to digits only.

    The agent may pass an order ID as ``#1053`` (copied from the customer
    message); tools must receive ``1053``. Customer-facing text is unaffected.
    """
    order_id = arguments.get("order_id")
    if order_id is None:
        return arguments
    normalized = _normalize_order_id(order_id)
    if normalized is None or normalized == order_id:
        return arguments
    return {**arguments, "order_id": normalized}


class AgentRun(BaseModel):
    """The structured result of one agent invocation."""

    run_id: str
    user_message: str
    model: str
    answer: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AgentEvent(BaseModel):
    """An incremental event emitted while an agent run is streaming.

    ``tool_call`` carries the call (with arguments) when it starts and the
    filled-in record when its result arrives; ``final`` carries the completed
    ``AgentRun``.
    """

    kind: Literal["tool_call", "tool_result", "final"]
    tool_call: ToolCallRecord | None = None
    run: AgentRun | None = None


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        if parts:
            return "".join(parts)
    return json.dumps(content, default=str)


def serialize_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Serialize an agent message list into plain structured dictionaries."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {
            "type": message.type,
            "content": content_to_text(getattr(message, "content", "")),
        }
        if isinstance(message, AIMessage) and message.tool_calls:
            entry["tool_calls"] = [
                {
                    "name": call.get("name", ""),
                    "args": normalize_tool_arguments(dict(call.get("args") or {})),
                    "id": call.get("id"),
                }
                for call in message.tool_calls
            ]
        if isinstance(message, ToolMessage):
            entry["tool_call_id"] = message.tool_call_id
            entry["name"] = message.name
            entry["status"] = getattr(message, "status", None)
        serialized.append(entry)
    return serialized


def extract_tool_calls(messages: Sequence[BaseMessage]) -> list[ToolCallRecord]:
    """Pair ordered tool calls with their results from an agent message list."""
    results: dict[str, ToolMessage] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id:
            results[message.tool_call_id] = message

    records: list[ToolCallRecord] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            call_id = call.get("id")
            record = ToolCallRecord(
                tool_name=call.get("name", ""),
                call_id=call_id,
                arguments=normalize_tool_arguments(dict(call.get("args") or {})),
            )
            tool_message = results.get(call_id) if call_id else None
            if tool_message is not None:
                text = content_to_text(tool_message.content)
                if getattr(tool_message, "status", "success") == "error":
                    record.error = text
                else:
                    record.result = text
            records.append(record)
    return records


def extract_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the last assistant text that is not a tool call."""
    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and message.content
        ):
            return content_to_text(message.content)
    return ""
