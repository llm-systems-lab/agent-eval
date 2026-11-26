from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from agent.behavior_checks import run_checks
from agent.config import AgentSettings
from agent.evaluation_runner import (
    EvaluationRunner,
    ScenarioResult,
    main,
    summarize,
)
from agent.scenarios import Scenario, load_scenarios
from agent.traces import AgentRun, ToolCallRecord


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted assistant messages, one per model turn."""

    responses: list[AIMessage] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
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


class ErrorChatModel(BaseChatModel):
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


class SleepingChatModel(BaseChatModel):
    """Blocks in the model call so the per-scenario timeout can interrupt it."""

    delay: float = 10.0

    @property
    def _llm_type(self) -> str:
        return "sleeping"

    def bind_tools(self, tools: Any, **kwargs: Any) -> SleepingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])


def scripted_factory(responses: list[AIMessage]):
    def factory() -> ScriptedChatModel:
        return ScriptedChatModel(responses=list(responses))

    return factory


def make_runner(
    tmp_path: Path,
    factory,
    *,
    use_metrics: bool = False,
    metrics=None,
    judge=None,
) -> EvaluationRunner:
    return EvaluationRunner(
        results_dir=tmp_path,
        label="test",
        settings=AgentSettings(google_api_key="dummy", model="fake-model"),
        model_factory=factory,
        configure_tracing=False,
        use_metrics=use_metrics,
        metrics=metrics,
        judge=judge,
    )


class FakeJudge:
    def __init__(self, passed: bool = True, explanation: str = "ok", error=None) -> None:
        self.passed = passed
        self.explanation = explanation
        self.error = error

    def generate(self, prompt, schema=None):
        from agent.judge_checks import JudgeVerdict

        if self.error is not None:
            raise RuntimeError(self.error)
        return JudgeVerdict(passed=self.passed, explanation=self.explanation)


class FakeMetric:
    def __init__(
        self, score: float = 0.9, success: bool = True, reason: str = "ok", error=None
    ) -> None:
        self.score = score
        self.success = success
        self.reason = reason
        self.error = error

    def measure(self, test_case) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)


def cancellation_scenario(scenario_id: str) -> Scenario:
    return Scenario(
        id=scenario_id,
        category="cancellation",
        customer_message="Please cancel order #1055.",
        order_id="1055",
        expected_tool_calls=["get_order_status", "cancel_order"],
        expected_facts=["The actual cancellation result must be reported."],
    )


def cancellation_001_scenario() -> Scenario:
    return Scenario(
        id="cancellation_001",
        category="cancellation",
        customer_message="Please cancel order #1055. I ordered the wrong cable.",
        order_id="1055",
        expected_tool_calls=["get_order_status", "cancel_order"],
        should_escalate=False,
        expected_facts=[
            "The agent must check the order status before cancellation.",
            "Cancellation may be attempted only if the status is processing.",
            "The agent must report the actual cancellation result.",
        ],
        forbidden_behaviors=[
            "Calling cancel_order before checking the status.",
            "Claiming cancellation succeeded without a successful tool result.",
        ],
    )


def cancel_then_final() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_order_status", "args": {"order_id": "1055"}, "id": "c1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "cancel_order", "args": {"order_id": "1055"}, "id": "c2"}
            ],
        ),
        AIMessage(content="Order 1055 is cancelled."),
    ]


def result_json(scenario_id: str, status: str) -> str:
    return ScenarioResult(
        scenario_id=scenario_id,
        category="cancellation",
        customer_message="Please cancel order #1055.",
        order_id="1055",
        status=status,
    ).model_dump_json()


def write_results_file(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "test_20260101T000000_abc123.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestDataset:
    def test_loads_scenarios(self) -> None:
        scenarios = load_scenarios()

        assert scenarios
        assert all(scenario.id and scenario.category for scenario in scenarios)
        categories = {scenario.category for scenario in scenarios}
        assert categories <= {
            "order_status",
            "order_details",
            "refund",
            "cancellation",
            "escalation",
        }


class TestBehaviorChecks:
    def _run(self, tool_calls: list[ToolCallRecord]) -> AgentRun:
        return AgentRun(
            run_id="r", user_message="m", model="fake", answer="a", tool_calls=tool_calls
        )

    def test_tool_calls_pass_when_expected_present_in_order(self) -> None:
        scenario = cancellation_scenario("c1")
        run = self._run(
            [
                ToolCallRecord(tool_name="get_order_status", arguments={"order_id": "1055"}),
                ToolCallRecord(tool_name="cancel_order", arguments={"order_id": "1055"}),
            ]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["expected_tool_calls"].status == "pass"
        assert checks["tool_arguments"].status == "pass"
        assert checks["escalation"].status == "pass"
        assert "expected_facts" not in checks  # judge checks omitted without a judge

    def test_forbidden_tool_call_fails(self) -> None:
        scenario = Scenario(
            id="c4",
            category="cancellation",
            customer_message="Please cancel order #1043.",
            order_id="1043",
            expected_tool_calls=["get_order_status"],
            forbidden_behaviors=["Calling cancel_order for a shipped order."],
        )
        run = self._run(
            [
                ToolCallRecord(tool_name="get_order_status", arguments={"order_id": "1043"}),
                ToolCallRecord(tool_name="cancel_order", arguments={"order_id": "1043"}),
            ]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["expected_tool_calls"].status == "fail"

    def test_cancellation_001_expected_cancel_is_not_forbidden(self) -> None:
        scenario = cancellation_001_scenario()
        run = self._run(
            [
                ToolCallRecord(
                    tool_name="get_order_status",
                    arguments={"order_id": "1055"},
                    result='{"success": true, "status": "processing"}',
                ),
                ToolCallRecord(
                    tool_name="cancel_order",
                    arguments={"order_id": "1055"},
                    result='{"success": true, "cancelled": true}',
                ),
            ]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["expected_tool_calls"].status == "pass"
        assert checks["cancellation_precondition"].status == "pass"

    def test_cancellation_precondition_fails_when_cancel_first(self) -> None:
        scenario = cancellation_001_scenario()
        run = self._run(
            [
                ToolCallRecord(
                    tool_name="cancel_order",
                    arguments={"order_id": "1055"},
                    result='{"success": true, "cancelled": true}',
                )
            ]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["cancellation_precondition"].status == "fail"

    def test_cancellation_precondition_fails_when_status_not_processing(self) -> None:
        scenario = cancellation_001_scenario()
        run = self._run(
            [
                ToolCallRecord(
                    tool_name="get_order_status",
                    arguments={"order_id": "1055"},
                    result='{"success": true, "status": "shipped"}',
                ),
                ToolCallRecord(
                    tool_name="cancel_order",
                    arguments={"order_id": "1055"},
                    result='{"success": false}',
                ),
            ]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["cancellation_precondition"].status == "fail"

    def test_missing_expected_tool_call_fails(self) -> None:
        scenario = cancellation_scenario("c2")
        run = self._run(
            [ToolCallRecord(tool_name="get_order_status", arguments={"order_id": "1055"})]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["expected_tool_calls"].status == "fail"

    def test_wrong_order_id_fails_argument_check(self) -> None:
        scenario = cancellation_scenario("c3")
        run = self._run(
            [ToolCallRecord(tool_name="get_order_status", arguments={"order_id": "9999"})]
        )

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["tool_arguments"].status == "fail"

    def test_escalation_check_detects_missing_escalation(self) -> None:
        scenario = Scenario(
            id="e1",
            category="escalation",
            customer_message="I want a human.",
            order_id="1063",
            expected_tool_calls=["escalate_to_human"],
            should_escalate=True,
        )
        run = self._run([])

        checks = {c.name: c for c in run_checks(scenario, run)}

        assert checks["escalation"].status == "fail"


class TestRunner:
    def test_run_executes_and_writes_incremental_results(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory(cancel_then_final()))

        result = runner.run([cancellation_scenario("s1"), cancellation_scenario("s2")])

        assert result.summary.total == 2
        assert result.summary.successful == 2
        assert result.summary.failed == 0
        assert result.summary.scenarios_passed == 2

        lines = Path(result.results_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert Path(result.summary_path).is_file()

    def test_each_scenario_gets_an_isolated_store(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory(cancel_then_final()))

        result = runner.run([cancellation_scenario("a"), cancellation_scenario("b")])

        for scenario_result in result.results:
            cancel_records = [
                c for c in scenario_result.tool_calls if c.tool_name == "cancel_order"
            ]
            assert cancel_records, "cancel_order was not called"
            payload = json.loads(cancel_records[0].result or "{}")
            assert payload["cancelled"] is True, (
                "store state leaked between scenarios"
            )

    def test_api_error_is_recorded_and_execution_continues(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, ErrorChatModel)

        result = runner.run([cancellation_scenario("s1")])

        assert result.summary.failed == 1
        assert result.results[0].status == "error"
        assert "simulated API failure" in (result.results[0].error or "")

    def test_run_label_is_sanitized(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="hi")]))
        runner.label = "../../evil"

        result = runner.run([cancellation_scenario("s1")])

        assert Path(result.results_path).parent == tmp_path
        assert Path(result.summary_path).parent == tmp_path

    def test_run_ids_are_unique(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="hi")]))

        first = runner.run([cancellation_scenario("s1")])
        second = runner.run([cancellation_scenario("s2")])

        assert first.results_path != second.results_path

    def test_progress_hooks_report_each_scenario(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="hi")]))
        starts: list[tuple[str, int, int]] = []
        results: list[tuple[str, int, int]] = []

        runner.run(
            [cancellation_scenario("s1"), cancellation_scenario("s2")],
            on_start=lambda s, i, t: starts.append((s.id, i, t)),
            on_result=lambda r, i, t: results.append((r.scenario_id, i, t)),
        )

        assert starts == [("s1", 1, 2), ("s2", 2, 2)]
        assert results == [("s1", 1, 2), ("s2", 2, 2)]

    def test_build_agent_sets_scenario_metadata(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        agent = runner._build_agent(cancellation_scenario("s7"))

        assert agent.metadata["scenario_id"] == "s7"
        assert agent.metadata["scenario_category"] == "cancellation"
        assert agent.metadata["experiment"] == "baseline"

    def test_result_records_agent_model_and_duration(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        result = runner.run([cancellation_scenario("s1")])

        row = result.results[0]
        assert row.agent_model == "fake-model"
        assert row.duration_ms is not None
        assert row.duration_ms >= 0
        persisted = json.loads(Path(result.results_path).read_text().strip())
        assert persisted["agent_model"] == "fake-model"
        assert persisted["duration_ms"] is not None

    def test_runner_attaches_and_persists_metric_scores(self, tmp_path: Path) -> None:
        metrics = [
            ("Answer Relevancy", FakeMetric(score=0.75)),
            ("Faithfulness", FakeMetric(score=1.0)),
        ]
        runner = make_runner(
            tmp_path,
            scripted_factory([AIMessage(content="ok")]),
            use_metrics=True,
            metrics=metrics,
        )

        result = runner.run([cancellation_scenario("s1")])

        row = result.results[0]
        assert [m.name for m in row.metrics] == ["Answer Relevancy", "Faithfulness"]
        assert row.metrics[0].score == 0.75
        persisted = json.loads(Path(result.results_path).read_text().strip())
        assert persisted["metrics"][1]["score"] == 1.0

    def test_metric_error_does_not_stop_the_run(self, tmp_path: Path) -> None:
        metrics = [
            ("Answer Relevancy", FakeMetric(error="rate limited")),
            ("Faithfulness", FakeMetric(score=0.6)),
        ]
        runner = make_runner(
            tmp_path,
            scripted_factory([AIMessage(content="ok")]),
            use_metrics=True,
            metrics=metrics,
        )

        result = runner.run([cancellation_scenario("s1"), cancellation_scenario("s2")])

        assert result.summary.successful == 2
        first = result.results[0].metrics
        assert first[0].error is not None
        assert first[1].score == 0.6
        assert result.results[1].metrics[1].score == 0.6

    def test_runner_records_judge_check_errors_and_continues(
        self, tmp_path: Path
    ) -> None:
        runner = make_runner(
            tmp_path,
            scripted_factory([AIMessage(content="ok")]),
            use_metrics=True,
            metrics=[],
            judge=FakeJudge(error="judge down"),
        )

        result = runner.run([cancellation_scenario("s1")])

        row = result.results[0]
        judge_checks = {
            c.name: c
            for c in row.checks
            if c.name in ("expected_facts", "forbidden_behaviors")
        }
        assert set(judge_checks) == {"expected_facts", "forbidden_behaviors"}
        assert judge_checks["expected_facts"].status == "error"
        assert "judge down" in judge_checks["expected_facts"].detail
        assert judge_checks["forbidden_behaviors"].status == "pass"
        assert result.summary.checks_error == 1
        assert row.status == "ok"

    def test_runner_records_judge_pass_and_fail(self, tmp_path: Path) -> None:
        scenario = Scenario(
            id="s1",
            category="refund",
            customer_message="Can I return order 1050?",
            order_id="1050",
            expected_facts=["Returns use a 30-day window."],
            forbidden_behaviors=["Promising a refund."],
        )
        runner = make_runner(
            tmp_path,
            scripted_factory([AIMessage(content="ok")]),
            use_metrics=True,
            metrics=[],
            judge=FakeJudge(passed=False, explanation="promised a refund"),
        )

        result = runner.run([scenario])

        checks = {c.name: c.status for c in result.results[0].checks}
        assert checks["expected_facts"] == "fail"
        assert checks["forbidden_behaviors"] == "fail"
        assert result.summary.checks_failed == 2

    def test_summary_reports_per_category(self) -> None:
        results = [
            ScenarioResult(
                scenario_id="a", category="refund", customer_message="x", status="ok"
            ),
            ScenarioResult(
                scenario_id="b", category="refund", customer_message="y", status="error"
            ),
        ]

        summary = summarize(results)

        assert summary.total == 2
        assert summary.per_category["refund"].total == 2
        assert summary.per_category["refund"].successful == 1
        assert summary.per_category["refund"].failed == 1

    def test_summary_counts_metrics(self) -> None:
        from agent.metric_evaluation import MetricScore

        results = [
            ScenarioResult(
                scenario_id="a",
                category="refund",
                customer_message="x",
                status="ok",
                metrics=[
                    MetricScore(name="pass", score=1.0, success=True),
                    MetricScore(name="fail", score=0.0, success=False),
                    MetricScore(name="error", error="boom"),
                ],
            )
        ]

        summary = summarize(results)

        assert summary.metrics_passed == 1
        assert summary.metrics_failed == 1
        assert summary.metrics_error == 1


class TestPerScenarioTimeout:
    def test_timeout_records_error_and_continues(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def factory() -> BaseChatModel:
            calls.append(1)
            if len(calls) == 1:
                return SleepingChatModel(delay=10.0)
            return ScriptedChatModel(responses=[AIMessage(content="ok")])

        runner = make_runner(tmp_path, factory)
        runner.scenario_timeout = 0.3

        started = time.monotonic()
        result = runner.run([cancellation_scenario("t1"), cancellation_scenario("t2")])
        elapsed = time.monotonic() - started

        assert result.results[0].status == "error"
        assert "timeout" in (result.results[0].error or "").lower()
        assert result.results[1].status == "ok"
        assert result.summary.failed == 1
        assert result.summary.successful == 1
        assert len(calls) == 2
        assert elapsed < 5.0, "the timeout did not interrupt the blocking call"

    def test_no_timeout_when_disabled(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))
        runner.scenario_timeout = None

        result = runner.run([cancellation_scenario("t1")])

        assert result.results[0].status == "ok"


class TestResume:
    def test_resume_skips_completed_and_retries_errors(self, tmp_path: Path) -> None:
        path = write_results_file(
            tmp_path, [result_json("s1", "ok"), result_json("s2", "error")]
        )
        calls: list[int] = []

        def factory() -> BaseChatModel:
            calls.append(1)
            return ScriptedChatModel(responses=[AIMessage(content="ok")])

        runner = make_runner(tmp_path, factory)

        result = runner.resume(
            path, [cancellation_scenario("s1"), cancellation_scenario("s2")]
        )

        assert result.label == "test"
        assert result.run_id == "20260101T000000_abc123"
        assert result.results_path == str(path)
        assert len(calls) == 1, "only the errored scenario should be re-executed"
        statuses = {item.scenario_id: item.status for item in result.results}
        assert statuses == {"s1": "ok", "s2": "ok"}
        assert result.summary.successful == 2

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3, "the retried result should be appended"
        assert list(tmp_path.glob("*.jsonl")) == [path]
        assert (tmp_path / "test_20260101T000000_abc123.summary.json").is_file()

    def test_resume_runs_nothing_when_all_completed(self, tmp_path: Path) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])
        calls: list[int] = []

        def factory() -> BaseChatModel:
            calls.append(1)
            return ScriptedChatModel(responses=[AIMessage(content="ok")])

        runner = make_runner(tmp_path, factory)

        result = runner.resume(path, [cancellation_scenario("s1")])

        assert calls == []
        assert result.summary.successful == 1
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1

    def test_resume_keeps_missing_scenarios_out_of_existing_results(
        self, tmp_path: Path
    ) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])

        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        result = runner.resume(
            path, [cancellation_scenario("s1"), cancellation_scenario("s3")]
        )

        assert [item.scenario_id for item in result.results] == ["s1", "s3"]
        assert result.summary.total == 2

    def test_resume_missing_file_raises(self, tmp_path: Path) -> None:
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        with pytest.raises(FileNotFoundError):
            runner.resume(tmp_path / "nope_20260101T000000_abc123.jsonl")


class TestRerunScenario:
    def test_rerun_replaces_errored_scenario_with_success(self, tmp_path: Path) -> None:
        path = write_results_file(
            tmp_path, [result_json("s1", "ok"), result_json("s2", "error")]
        )
        runner = make_runner(
            tmp_path, scripted_factory([AIMessage(content="Order 1055 is cancelled.")])
        )

        result = runner.rerun_scenario(
            path, "s2", [cancellation_scenario("s1"), cancellation_scenario("s2")]
        )

        assert result.label == "test"
        assert result.run_id == "20260101T000000_abc123"
        assert [item.scenario_id for item in result.results] == ["s1", "s2"]
        assert result.results[1].status == "ok"
        assert result.summary.total == 2
        assert result.summary.successful == 2
        assert result.summary.failed == 0

    def test_rerun_preserves_other_scenario_records(self, tmp_path: Path) -> None:
        path = write_results_file(
            tmp_path, [result_json("s1", "ok"), result_json("s2", "error")]
        )
        before = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        result = runner.rerun_scenario(
            path, "s2", [cancellation_scenario("s1"), cancellation_scenario("s2")]
        )

        after = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert after[0] == before[0]
        assert after[1]["scenario_id"] == "s2"
        assert after[1]["status"] == "ok"
        assert result.results[0].model_dump() == ScenarioResult.model_validate(
            before[0]
        ).model_dump()

    def test_rerun_updates_summary_from_complete_result_set(
        self, tmp_path: Path
    ) -> None:
        path = write_results_file(
            tmp_path, [result_json("s1", "ok"), result_json("s2", "error")]
        )
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        result = runner.rerun_scenario(
            path, "s2", [cancellation_scenario("s1"), cancellation_scenario("s2")]
        )

        summary_path = Path(result.summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["total"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["per_category"]["cancellation"]["successful"] == 2
        assert summary["per_category"]["cancellation"]["failed"] == 0

    def test_rerun_rejects_unknown_dataset_scenario(self, tmp_path: Path) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        with pytest.raises(ValueError, match="not part of the dataset"):
            runner.rerun_scenario(path, "missing", [cancellation_scenario("s1")])

    def test_rerun_rejects_scenario_without_existing_record(
        self, tmp_path: Path
    ) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        with pytest.raises(ValueError, match="no existing record"):
            runner.rerun_scenario(
                path, "s2", [cancellation_scenario("s1"), cancellation_scenario("s2")]
            )

    def test_rerun_leaves_no_duplicate_scenario_records(self, tmp_path: Path) -> None:
        path = write_results_file(
            tmp_path,
            [
                result_json("s1", "ok"),
                result_json("s2", "error"),
                result_json("s2", "error"),
            ],
        )
        runner = make_runner(tmp_path, scripted_factory([AIMessage(content="ok")]))

        result = runner.rerun_scenario(
            path, "s2", [cancellation_scenario("s1"), cancellation_scenario("s2")]
        )

        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [json.loads(line)["scenario_id"] for line in lines]
        assert ids == ["s1", "s2"]
        assert len(result.results) == 2
        assert len({item.scenario_id for item in result.results}) == 2


class TestEvaluationCli:
    def test_scenario_id_requires_resume(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--scenario-id", "s1", "--no-metrics"])

        assert excinfo.value.code == 2

    def test_cli_rejects_unknown_scenario_id(self, tmp_path: Path) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])
        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "id": "s1",
                            "category": "cancellation",
                            "customer_message": "Please cancel order #1055.",
                            "order_id": "1055",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        code = main(
            [
                "--scenario-id",
                "missing",
                "--resume",
                str(path),
                "--dataset",
                str(dataset),
                "--no-metrics",
            ]
        )

        assert code == 1

    def test_cli_rejects_scenario_without_existing_record(
        self, tmp_path: Path
    ) -> None:
        path = write_results_file(tmp_path, [result_json("s1", "ok")])
        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "id": "s1",
                            "category": "cancellation",
                            "customer_message": "Please cancel order #1055.",
                            "order_id": "1055",
                        },
                        {
                            "id": "s2",
                            "category": "cancellation",
                            "customer_message": "Please cancel order #1055.",
                            "order_id": "1055",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        code = main(
            [
                "--scenario-id",
                "s2",
                "--resume",
                str(path),
                "--dataset",
                str(dataset),
                "--no-metrics",
            ]
        )

        assert code == 1
