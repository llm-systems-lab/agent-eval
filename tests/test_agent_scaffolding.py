from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from agent.baseline_agent import (
    BASELINE_TAG,
    SYSTEM_PROMPT,
    BaselineAgent,
    create_baseline_agent,
)
from agent.config import (
    DEFAULT_OPENROUTER_BASE_URL,
    AgentSettings,
    load_agent_settings,
    load_judge_settings,
    load_langsmith_settings,
)
from agent.toolkit import build_customer_support_tools
from agent.tools import DEFAULT_DATA_PATH, EscalationRegistry, OrderStore
from agent.tools import customer_support_tools as cst
from agent.traces import (
    extract_answer,
    extract_tool_calls,
    normalize_tool_arguments,
    serialize_messages,
)

RAW_STORE = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
REFERENCE_DATE = date(2026, 11, 24)


def make_store() -> OrderStore:
    return OrderStore(data=RAW_STORE)


def write_env(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(contents, encoding="utf-8")
    return path


class TestConfig:
    def test_agent_settings_loaded_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = write_env(
            tmp_path,
            "GOOGLE_API_KEY=agent-key\nAGENT_MODEL=gemini-test\nAGENT_TEMPERATURE=0.5\n",
        )
        for name in ("GOOGLE_API_KEY", "AGENT_MODEL", "AGENT_TEMPERATURE"):
            monkeypatch.delenv(name, raising=False)

        settings = load_agent_settings(env_path=env)

        assert settings.model == "gemini-test"
        assert settings.temperature == 0.5
        assert settings.google_api_key.get_secret_value() == "agent-key"
        assert "agent-key" not in repr(settings)

    def test_agent_settings_missing_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = write_env(tmp_path, "")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_MODEL", raising=False)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            load_agent_settings(env_path=env)

    def test_judge_settings_are_independent_of_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = write_env(
            tmp_path,
            "OPENROUTER_API_KEY=or-key\nJUDGE_MODEL=groq/model\n",
        )
        for name in (
            "OPENROUTER_API_KEY",
            "JUDGE_MODEL",
            "OPENROUTER_BASE_URL",
            "GOOGLE_API_KEY",
            "AGENT_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

        settings = load_judge_settings(env_path=env)

        assert settings.model == "groq/model"
        assert settings.base_url == DEFAULT_OPENROUTER_BASE_URL
        assert settings.openrouter_api_key.get_secret_value() == "or-key"

    def test_langsmith_settings_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = write_env(
            tmp_path,
            "LANGSMITH_TRACING=true\nLANGSMITH_PROJECT=proj\nLANGSMITH_API_KEY=ls-key\n",
        )
        for name in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        settings = load_langsmith_settings(env_path=env)

        assert settings.tracing is True
        assert settings.project == "proj"
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "ls-key"


class TestTraces:
    def test_extracts_ordered_tool_calls_with_results(self) -> None:
        messages = [
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
            ToolMessage(
                content='{"success": true}',
                tool_call_id="call-1",
                name="get_order_status",
            ),
            AIMessage(content="Order 1042 is processing."),
        ]

        records = extract_tool_calls(messages)

        assert len(records) == 1
        assert records[0].tool_name == "get_order_status"
        assert records[0].call_id == "call-1"
        assert records[0].arguments == {"order_id": "1042"}
        assert records[0].result == '{"success": true}'
        assert records[0].error is None
        assert extract_answer(messages) == "Order 1042 is processing."

    def test_extracts_text_from_content_blocks(self) -> None:
        message = AIMessage(
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
                {"type": "tool_call", "name": "t", "args": {}, "id": "x"},
            ]
        )

        assert extract_answer([message]) == "Hello world"

    def test_serializes_messages_into_structured_dicts(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "1"}, "id": "c"}
                ],
            ),
            ToolMessage(
                content='{"success": true}',
                tool_call_id="c",
                name="get_order_status",
            ),
            AIMessage(content="Done."),
        ]

        serialized = serialize_messages(messages)

        assert serialized[0]["tool_calls"][0]["name"] == "get_order_status"
        assert serialized[1]["tool_call_id"] == "c"
        assert serialized[1]["name"] == "get_order_status"
        assert serialized[2]["content"] == "Done."

    def test_tool_error_is_captured(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "cancel_order", "args": {"order_id": "1"}, "id": "c"}
                ],
            ),
            ToolMessage(
                content="boom",
                tool_call_id="c",
                name="cancel_order",
                status="error",
            ),
        ]

        records = extract_tool_calls(messages)

        assert records[0].error == "boom"
        assert records[0].result is None

    def test_normalize_tool_arguments_strips_leading_hash(self) -> None:
        assert normalize_tool_arguments({"order_id": "#1053"}) == {"order_id": "1053"}
        assert normalize_tool_arguments({"order_id": "1053"}) == {"order_id": "1053"}
        assert normalize_tool_arguments({"order_id": " #1053 "}) == {
            "order_id": "1053"
        }

    def test_normalize_tool_arguments_leaves_other_arguments(self) -> None:
        assert normalize_tool_arguments({"reason": "x"}) == {"reason": "x"}
        assert normalize_tool_arguments({"order_id": None}) == {"order_id": None}

    def test_extract_tool_calls_normalizes_order_id(self) -> None:
        with_hash = extract_tool_calls(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "#1042"},
                            "id": "c",
                        }
                    ],
                )
            ]
        )
        without_hash = extract_tool_calls(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "1042"},
                            "id": "c",
                        }
                    ],
                )
            ]
        )

        assert with_hash[0].arguments["order_id"] == "1042"
        assert without_hash[0].arguments["order_id"] == "1042"

    def test_serialize_messages_normalizes_order_id(self) -> None:
        serialized = serialize_messages(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "#1042"},
                            "id": "c",
                        }
                    ],
                )
            ]
        )

        assert serialized[0]["tool_calls"][0]["args"]["order_id"] == "1042"


class TestToolkit:
    def test_exposes_expected_tool_names(self) -> None:
        tools = build_customer_support_tools(store=make_store())

        assert [tool.name for tool in tools] == [
            "get_order_status",
            "get_order_details",
            "check_refund_eligibility",
            "cancel_order",
            "escalate_to_human",
        ]

    def test_delegates_to_existing_tool_logic(self) -> None:
        tools = {t.name: t for t in build_customer_support_tools(store=make_store())}

        result = json.loads(tools["get_order_status"].invoke({"order_id": "1042"}))

        assert result["success"] is True
        assert result["status"] == "processing"

    def test_toolkit_normalizes_order_id_before_calling_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        real = cst.get_order_status

        def spy(order_id: str, *, store: OrderStore | None = None) -> object:
            captured["order_id"] = order_id
            return real(order_id, store=store)

        monkeypatch.setattr(cst, "get_order_status", spy)
        tools = {t.name: t for t in build_customer_support_tools(store=make_store())}

        result = json.loads(tools["get_order_status"].invoke({"order_id": "#1042"}))

        assert captured["order_id"] == "1042"
        assert result["status"] == "processing"

    def test_toolkit_normalizes_escalation_order_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        real = cst.escalate_to_human

        def spy(
            order_id: str | None,
            reason: str,
            *,
            registry: EscalationRegistry | None = None,
        ) -> object:
            captured["order_id"] = order_id
            return real(order_id, reason, registry=registry)

        monkeypatch.setattr(cst, "escalate_to_human", spy)
        tools = {
            t.name: t
            for t in build_customer_support_tools(
                store=make_store(), registry=EscalationRegistry()
            )
        }

        tools["escalate_to_human"].invoke(
            {"reason": "damaged item", "order_id": "#1053"}
        )

        assert captured["order_id"] == "1053"

    def test_cancel_mutates_injected_store(self) -> None:
        store = make_store()
        tools = {t.name: t for t in build_customer_support_tools(store=store)}

        result = json.loads(tools["cancel_order"].invoke({"order_id": "1055"}))

        assert result["cancelled"] is True
        assert store.get_order("1055").status == "cancelled"

    def test_refund_uses_bound_reference_date(self) -> None:
        tools = {
            t.name: t
            for t in build_customer_support_tools(
                store=make_store(), reference_date=REFERENCE_DATE
            )
        }

        result = json.loads(
            tools["check_refund_eligibility"].invoke({"order_id": "1050"})
        )

        assert result["eligible"] is True
        assert result["days_since_delivery"] == 10

    def test_escalation_uses_injected_registry(self) -> None:
        registry = EscalationRegistry()
        tools = {
            t.name: t
            for t in build_customer_support_tools(store=make_store(), registry=registry)
        }

        result = json.loads(
            tools["escalate_to_human"].invoke(
                {"order_id": "1053", "reason": "damaged"}
            )
        )

        assert result["success"] is True
        assert len(registry.list_escalations()) == 1

    def test_escalation_requires_reason(self) -> None:
        tools = {
            t.name: t
            for t in build_customer_support_tools(
                store=make_store(), registry=EscalationRegistry()
            )
        }

        with pytest.raises(ValidationError):
            tools["escalate_to_human"].invoke({"order_id": "1053"})

    def test_escalate_tool_description_states_conditions(self) -> None:
        tools = {t.name: t for t in build_customer_support_tools(store=make_store())}

        description = (tools["escalate_to_human"].description or "").lower()

        for keyword in (
            "cannot be found",
            "delivery delay",
            "damaged",
            "human representative",
            "unresolved",
        ):
            assert keyword in description


class TestAgentWiring:
    def test_system_prompt_states_escalation_rules(self) -> None:
        prompt = SYSTEM_PROMPT.lower()

        assert "escalate_to_human" in SYSTEM_PROMPT
        assert "permission" in prompt
        for keyword in (
            "cannot be found",
            "delivery delay",
            "damaged, missing, tampered",
            "human representative",
            "unresolved",
        ):
            assert keyword in prompt

    def test_system_prompt_states_refund_policy(self) -> None:
        prompt = SYSTEM_PROMPT.lower()

        assert "eligibility check" in prompt
        assert "refund has been processed" in prompt
        assert "refund-processing tool" in prompt

    def test_order_status_prompt_forbids_invented_times(self) -> None:
        prompt = " ".join(SYSTEM_PROMPT.lower().split())

        assert (
            "do not invent order information, response times, or follow-up details"
            in prompt
        )
        assert (
            "never promise a callback, a response time, or any follow-up timeframe"
            in prompt
        )

    def test_damaged_item_refund_prompt_forbids_callback_promises(self) -> None:
        prompt = " ".join(SYSTEM_PROMPT.lower().split())

        assert "never promise a callback" in prompt
        assert "unless a tool result explicitly confirms it" in prompt
        assert "do not invent what happens next or when someone will respond" in prompt
        assert (
            "do not add a promise about when or how the support team will follow up"
            in prompt
        )

    def test_human_escalation_prompt_allows_confirmation_with_reference(self) -> None:
        prompt = " ".join(SYSTEM_PROMPT.lower().split())

        assert "when escalate_to_human succeeds" in prompt
        assert "confirm that the case was escalated" in prompt
        assert "reference number" in prompt
        assert (
            "your case has been escalated to our support team. "
            "your reference number is esc-000001."
        ) in prompt

    def _agent(self) -> BaselineAgent:
        settings = AgentSettings(google_api_key="dummy-key", model="gemini-test")
        return create_baseline_agent(
            settings=settings,
            store=make_store(),
            configure_tracing=False,
        )

    def test_create_baseline_agent_builds_without_network(self) -> None:
        agent = self._agent()

        assert agent.model_name == "gemini-test"

    def test_run_config_carries_baseline_tag_and_metadata(self) -> None:
        agent = self._agent()

        config = agent.run_config

        assert BASELINE_TAG in config["tags"]
        assert config["metadata"]["experiment"] == "baseline"
        assert config["metadata"]["agent_version"] == 1
        assert config["recursion_limit"] == agent.recursion_limit
        assert config["run_name"]

    def test_custom_metadata_merges_with_baseline_defaults(self) -> None:
        settings = AgentSettings(google_api_key="dummy-key", model="gemini-test")
        agent = create_baseline_agent(
            settings=settings,
            store=make_store(),
            configure_tracing=False,
            metadata={"scenario_id": "order_status_001"},
        )

        assert agent.metadata["scenario_id"] == "order_status_001"
        assert agent.metadata["experiment"] == "baseline"
        assert agent.metadata["agent_version"] == 1
