"""Regression comparison between a candidate evaluation run and a baseline.

This module reads results already persisted by the evaluation runner. It does
not execute scenarios, call the agent, or invoke metrics. A regression is a
scenario, deterministic check, or metric that passed in the baseline but no
longer passes in the candidate. Candidate metric errors are reported separately
and are never treated as passes. A configurable score tolerance ignores small
drops between two passing metric results.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from .evaluation_runner import ScenarioResult, _read_results

DEFAULT_SCORE_TOLERANCE = 0.1


class RegressionIssue(BaseModel):
    """One comparison finding."""

    scenario_id: str
    kind: str
    name: str
    detail: str
    baseline_score: float | None = None
    candidate_score: float | None = None
    threshold: float | None = None


class RegressionReport(BaseModel):
    """Outcome of comparing a candidate run against a baseline run."""

    baseline_path: str = ""
    candidate_path: str = ""
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE
    baseline_scenarios: int = 0
    candidate_scenarios: int = 0
    compared_scenarios: int = 0
    regressions: list[RegressionIssue] = Field(default_factory=list)
    metric_errors: list[RegressionIssue] = Field(default_factory=list)
    new_failures: list[RegressionIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A run passes only when no regression was detected."""
        return not self.regressions


def _metric_state(metric: object | None) -> str:
    """Classify a metric record as pass, fail, error, or missing.

    A missing or errored metric is never classified as a pass. When ``success``
    is absent, the recorded score and threshold are used instead.
    """
    if metric is None:
        return "missing"
    if getattr(metric, "error", None) is not None:
        return "error"
    success = getattr(metric, "success", None)
    if success is True:
        return "pass"
    if success is False:
        return "fail"
    score = getattr(metric, "score", None)
    threshold = getattr(metric, "threshold", None)
    if score is not None and threshold is not None:
        return "pass" if score >= threshold else "fail"
    return "error"


def _metric_score(metric: object | None) -> float | None:
    if metric is None:
        return None
    return getattr(metric, "score", None)


def _metric_threshold(metric: object | None) -> float | None:
    if metric is None:
        return None
    return getattr(metric, "threshold", None)


def compare(
    baseline: Sequence[ScenarioResult],
    candidate: Sequence[ScenarioResult],
    *,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> RegressionReport:
    """Compare candidate results against baseline results.

    Only scenarios and metrics present in the baseline are compared. A baseline
    pass that becomes a candidate failure, error, or missing record is a
    regression. A score drop greater than ``score_tolerance`` between two
    passing metric results is also a regression.
    """
    report = RegressionReport(
        score_tolerance=score_tolerance,
        baseline_scenarios=len(baseline),
        candidate_scenarios=len(candidate),
    )
    baseline_by_id = {result.scenario_id: result for result in baseline}
    candidate_by_id = {result.scenario_id: result for result in candidate}

    for scenario_id, base in baseline_by_id.items():
        report.compared_scenarios += 1
        cand = candidate_by_id.get(scenario_id)
        if cand is None:
            report.regressions.append(
                RegressionIssue(
                    scenario_id=scenario_id,
                    kind="scenario",
                    name="execution",
                    detail="scenario is missing from the candidate run",
                )
            )
            continue

        if base.status == "ok" and cand.status != "ok":
            error = f": {cand.error}" if cand.error else ""
            report.regressions.append(
                RegressionIssue(
                    scenario_id=scenario_id,
                    kind="scenario",
                    name="execution",
                    detail=f"baseline status ok; candidate status {cand.status}{error}",
                )
            )

        candidate_checks = {check.name: check for check in cand.checks}
        for base_check in base.checks:
            if base_check.status != "pass":
                continue
            candidate_check = candidate_checks.get(base_check.name)
            if candidate_check is None:
                report.regressions.append(
                    RegressionIssue(
                        scenario_id=scenario_id,
                        kind="check",
                        name=base_check.name,
                        detail="check is missing from the candidate run",
                    )
                )
            elif candidate_check.status != "pass":
                detail = f"baseline passed; candidate {candidate_check.status}"
                if candidate_check.detail:
                    detail = f"{detail}: {candidate_check.detail}"
                report.regressions.append(
                    RegressionIssue(
                        scenario_id=scenario_id,
                        kind="check",
                        name=base_check.name,
                        detail=detail,
                    )
                )

        candidate_metrics = {metric.name: metric for metric in cand.metrics}
        for base_metric in base.metrics:
            base_state = _metric_state(base_metric)
            candidate_metric = candidate_metrics.get(base_metric.name)
            candidate_state = _metric_state(candidate_metric)
            baseline_score = _metric_score(base_metric)
            candidate_score = _metric_score(candidate_metric)
            threshold = _metric_threshold(candidate_metric)
            if threshold is None:
                threshold = _metric_threshold(base_metric)

            if base_state == "pass":
                if candidate_state != "pass":
                    report.regressions.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail=f"baseline passed; candidate {candidate_state}",
                            baseline_score=baseline_score,
                            candidate_score=candidate_score,
                            threshold=threshold,
                        )
                    )
                elif (
                    baseline_score is not None
                    and candidate_score is not None
                    and baseline_score - candidate_score > score_tolerance
                ):
                    report.regressions.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail=(
                                f"baseline passed at {baseline_score:.2f}; candidate "
                                f"passed at {candidate_score:.2f}, a drop greater than "
                                f"{score_tolerance:g}"
                            ),
                            baseline_score=baseline_score,
                            candidate_score=candidate_score,
                            threshold=threshold,
                        )
                    )
            elif base_state == "error":
                if candidate_state == "error":
                    report.metric_errors.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail="baseline and candidate both errored",
                            baseline_score=baseline_score,
                            candidate_score=candidate_score,
                            threshold=threshold,
                        )
                    )
                elif candidate_state == "fail":
                    report.new_failures.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail="baseline errored; candidate failed",
                            baseline_score=baseline_score,
                            candidate_score=candidate_score,
                            threshold=threshold,
                        )
                    )
                elif candidate_state == "missing":
                    report.metric_errors.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail="baseline errored; metric is missing from the candidate run",
                            baseline_score=baseline_score,
                            threshold=threshold,
                        )
                    )
            elif base_state == "fail":
                if candidate_state == "error":
                    report.metric_errors.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail="baseline failed; candidate errored",
                            baseline_score=baseline_score,
                            candidate_score=candidate_score,
                            threshold=threshold,
                        )
                    )
                elif candidate_state == "missing":
                    report.metric_errors.append(
                        RegressionIssue(
                            scenario_id=scenario_id,
                            kind="metric",
                            name=base_metric.name,
                            detail="baseline failed; metric is missing from the candidate run",
                            baseline_score=baseline_score,
                            threshold=threshold,
                        )
                    )

    return report


def _resolve_jsonl(path: str | Path) -> Path:
    """Return a results JSONL file, or the newest one in a directory."""
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Results path not found: {resolved}")
    candidates = sorted(resolved.glob("*.jsonl"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No results JSONL files found in: {resolved}")
    return candidates[-1]


def _print_issues(console: Console, title: str, issues: Sequence[RegressionIssue]) -> None:
    if not issues:
        return
    console.print(f"{title}: {len(issues)}")
    for issue in issues:
        console.print(
            f"  - {issue.scenario_id} | {issue.kind} | {issue.name} | {issue.detail}"
        )


def _print_report(console: Console, report: RegressionReport) -> None:
    console.print(
        f"Baseline: {report.baseline_path} ({report.baseline_scenarios} scenarios)"
    )
    console.print(
        f"Candidate: {report.candidate_path} ({report.candidate_scenarios} scenarios)"
    )
    console.print(f"Score tolerance: {report.score_tolerance:g}")
    _print_issues(console, "Regressions", report.regressions)
    _print_issues(console, "Metric errors", report.metric_errors)
    _print_issues(console, "New failures (baseline had no valid result)", report.new_failures)
    if report.passed:
        console.print("[green]PASS[/green]: no regressions detected.")
    else:
        console.print(
            f"[red]FAIL[/red]: {len(report.regressions)} regression(s) detected."
        )


def main(argv: list[str] | None = None) -> int:
    """Compare a candidate results file against a baseline results file."""
    parser = argparse.ArgumentParser(
        description="Compare an evaluation run against a baseline run."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline results JSONL file or directory.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate results JSONL file or directory.",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=DEFAULT_SCORE_TOLERANCE,
        help=(
            "Allowed drop between two passing metric scores before it counts as "
            f"a regression (default {DEFAULT_SCORE_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--report",
        help="Write the comparison report as JSON to this path.",
    )
    args = parser.parse_args(argv)

    console = Console()
    try:
        baseline_path = _resolve_jsonl(args.baseline)
        candidate_path = _resolve_jsonl(args.candidate)
        baseline = _read_results(baseline_path)
        candidate = _read_results(candidate_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(
            Panel(str(exc), title="Regression comparison error", border_style="red")
        )
        return 2

    report = compare(baseline, candidate, score_tolerance=args.score_tolerance)
    report.baseline_path = str(baseline_path)
    report.candidate_path = str(candidate_path)
    _print_report(console, report)

    report_path = (
        Path(args.report)
        if args.report
        else candidate_path.with_suffix(".regression.json")
    )
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"Report: {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
