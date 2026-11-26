from __future__ import annotations

import json
import os
from pathlib import Path

from agent.behavior_checks import CheckResult
from agent.evaluation_runner import ScenarioResult
from agent.metric_evaluation import MetricScore
from agent.regression import _resolve_jsonl, compare, main


def make_result(
    scenario_id: str = "s1",
    *,
    status: str = "ok",
    checks: list[CheckResult] | None = None,
    metrics: list[MetricScore] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        category="refund",
        customer_message="hello",
        status=status,
        checks=checks or [],
        metrics=metrics or [],
    )


def passing_metric(name: str = "Answer Relevancy", score: float = 0.9) -> MetricScore:
    return MetricScore(name=name, score=score, success=True, threshold=0.5)


def failing_metric(name: str = "Answer Relevancy", score: float = 0.2) -> MetricScore:
    return MetricScore(name=name, score=score, success=False, threshold=0.5)


def errored_metric(name: str = "Answer Relevancy") -> MetricScore:
    return MetricScore(name=name, error="judge unavailable")


class TestCompare:
    def test_identical_runs_pass(self) -> None:
        base = make_result(metrics=[passing_metric()])
        candidate = make_result(metrics=[passing_metric()])

        report = compare([base], [candidate])

        assert report.passed is True
        assert report.regressions == []
        assert report.metric_errors == []
        assert report.new_failures == []

    def test_metric_pass_to_fail_is_regression(self) -> None:
        base = make_result(metrics=[passing_metric()])
        candidate = make_result(metrics=[failing_metric()])

        report = compare([base], [candidate])

        assert report.passed is False
        assert len(report.regressions) == 1
        issue = report.regressions[0]
        assert issue.kind == "metric"
        assert issue.name == "Answer Relevancy"
        assert "candidate fail" in issue.detail

    def test_metric_pass_to_error_is_regression(self) -> None:
        base = make_result(metrics=[passing_metric()])
        candidate = make_result(metrics=[errored_metric()])

        report = compare([base], [candidate])

        assert report.passed is False
        assert len(report.regressions) == 1
        assert "candidate error" in report.regressions[0].detail

    def test_missing_metric_is_regression(self) -> None:
        base = make_result(metrics=[passing_metric()])
        candidate = make_result(metrics=[])

        report = compare([base], [candidate])

        assert report.passed is False
        assert "candidate missing" in report.regressions[0].detail

    def test_score_drop_within_tolerance_passes(self) -> None:
        base = make_result(metrics=[passing_metric(score=0.9)])
        candidate = make_result(metrics=[passing_metric(score=0.85)])

        report = compare([base], [candidate], score_tolerance=0.1)

        assert report.passed is True
        assert report.regressions == []

    def test_score_drop_beyond_tolerance_is_regression(self) -> None:
        base = make_result(metrics=[passing_metric(score=0.9)])
        candidate = make_result(metrics=[passing_metric(score=0.7)])

        report = compare([base], [candidate], score_tolerance=0.1)

        assert report.passed is False
        assert "drop greater than 0.1" in report.regressions[0].detail

    def test_check_pass_to_fail_is_regression(self) -> None:
        base = make_result(
            checks=[CheckResult(name="expected_tool_calls", status="pass")]
        )
        candidate = make_result(
            checks=[CheckResult(name="expected_tool_calls", status="fail")]
        )

        report = compare([base], [candidate])

        assert report.passed is False
        assert report.regressions[0].kind == "check"
        assert report.regressions[0].name == "expected_tool_calls"

    def test_scenario_error_is_regression(self) -> None:
        base = make_result(status="ok")
        candidate = make_result(status="error")

        report = compare([base], [candidate])

        assert report.passed is False
        assert report.regressions[0].kind == "scenario"

    def test_missing_scenario_is_regression(self) -> None:
        report = compare([make_result("s1")], [])

        assert report.passed is False
        assert "missing from the candidate run" in report.regressions[0].detail

    def test_baseline_metric_error_is_reported_separately(self) -> None:
        base = make_result(metrics=[errored_metric()])
        candidate = make_result(metrics=[errored_metric()])

        report = compare([base], [candidate])

        assert report.passed is True
        assert report.regressions == []
        assert len(report.metric_errors) == 1
        assert "both errored" in report.metric_errors[0].detail

    def test_baseline_error_candidate_fail_is_new_failure(self) -> None:
        base = make_result(metrics=[errored_metric()])
        candidate = make_result(metrics=[failing_metric()])

        report = compare([base], [candidate])

        assert report.passed is True
        assert report.regressions == []
        assert len(report.new_failures) == 1

    def test_baseline_fail_candidate_pass_is_improvement(self) -> None:
        base = make_result(metrics=[failing_metric()])
        candidate = make_result(metrics=[passing_metric()])

        report = compare([base], [candidate])

        assert report.passed is True
        assert report.regressions == []


class TestCli:
    def _write(self, path: Path, results: list[ScenarioResult]) -> None:
        path.write_text(
            "\n".join(result.model_dump_json() for result in results) + "\n",
            encoding="utf-8",
        )

    def test_cli_passes_and_writes_report(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.jsonl"
        candidate = tmp_path / "candidate.jsonl"
        self._write(baseline, [make_result(metrics=[passing_metric()])])
        self._write(candidate, [make_result(metrics=[passing_metric()])])

        code = main(
            ["--baseline", str(baseline), "--candidate", str(candidate)]
        )

        assert code == 0
        report_path = candidate.with_suffix(".regression.json")
        assert report_path.is_file()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["regressions"] == []

    def test_cli_regression_returns_one(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.jsonl"
        candidate = tmp_path / "candidate.jsonl"
        self._write(baseline, [make_result(metrics=[passing_metric()])])
        self._write(candidate, [make_result(metrics=[failing_metric()])])

        code = main(
            ["--baseline", str(baseline), "--candidate", str(candidate)]
        )

        assert code == 1

    def test_cli_missing_baseline_returns_two(self, tmp_path: Path) -> None:
        candidate = tmp_path / "candidate.jsonl"
        self._write(candidate, [make_result()])

        code = main(
            [
                "--baseline",
                str(tmp_path / "missing.jsonl"),
                "--candidate",
                str(candidate),
            ]
        )

        assert code == 2

    def test_resolve_jsonl_picks_newest_file(self, tmp_path: Path) -> None:
        older = tmp_path / "older.jsonl"
        newer = tmp_path / "newer.jsonl"
        older.write_text("", encoding="utf-8")
        newer.write_text("", encoding="utf-8")
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))

        assert _resolve_jsonl(tmp_path) == newer
