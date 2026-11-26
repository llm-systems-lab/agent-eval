"""Evaluation runner for the Baseline Agent.

The runner loads the fixed dataset, executes scenarios sequentially against the
existing Baseline Agent, runs deterministic behavior checks over each trace, and
saves results incrementally so completed scenarios survive an interruption. It
is independent of the agent implementation and of the CLI.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .baseline_agent import create_baseline_agent
from .behavior_checks import CheckResult, run_checks
from .config import AgentSettings
from .metric_evaluation import MetricScore, evaluate_scenario
from .scenarios import Scenario, load_scenarios
from .tools import EscalationRegistry, OrderStore
from .traces import ToolCallRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MOCK_STORE_PATH = PROJECT_ROOT / "data" / "mock_store.json"
DEFAULT_REFERENCE_DATE = date(2026, 11, 24)
DEFAULT_SCENARIO_TIMEOUT = 60.0
_UNSAFE_LABEL_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_RUN_FILENAME = re.compile(r"^(?P<label>.+)_(?P<run_id>\d{8}T\d{6}_[0-9a-f]{6})$")


def _sanitize_label(label: str) -> str:
    """Keep the run label safe to use as a filename component."""
    sanitized = _UNSAFE_LABEL_CHARS.sub("_", label).strip("._") or "run"
    return sanitized


def _parse_run_filename(path: Path) -> tuple[str, str]:
    """Recover the label and run ID from a ``<label>_<run_id>.jsonl`` filename."""
    match = _RUN_FILENAME.match(path.stem)
    if match is None:
        raise ValueError(
            f"Cannot recover the run label and ID from filename: {path.name}"
        )
    return match.group("label"), match.group("run_id")


def _read_results(path: Path) -> list[ScenarioResult]:
    """Read the results already persisted in a run's JSONL file."""
    results: list[ScenarioResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            results.append(ScenarioResult.model_validate_json(stripped))
    return results


def _write_results(path: Path, results: Sequence[ScenarioResult]) -> None:
    """Rewrite a results JSONL file atomically with exactly the given records."""
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        for result in results:
            file.write(result.model_dump_json() + "\n")
        temp_path = Path(file.name)
    os.replace(temp_path, path)


class ScenarioTimeoutError(Exception):
    """Raised when a scenario exceeds its per-scenario timeout."""


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _elapsed_ms(started_at: datetime, finished_at: datetime) -> int:
    return int((finished_at - started_at).total_seconds() * 1000)


def _run_with_timeout(seconds: float | None, func: Callable[[], Any]) -> Any:
    """Run ``func`` and interrupt it if it exceeds ``seconds``.

    On POSIX main-thread execution a ``SIGALRM`` timer raises inside the
    blocking call, so the wait is genuinely interrupted and no worker is left
    running. Where signals are unavailable (other platforms or a non-main
    thread) the call runs in a worker thread and is abandoned on timeout.
    """
    if not seconds or seconds <= 0:
        return func()

    if hasattr(signal, "SIGALRM") and _is_main_thread():
        def _handler(signum: int, frame: Any) -> None:
            raise ScenarioTimeoutError(f"scenario exceeded {seconds:g}s timeout")

        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return func()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError as exc:
        raise ScenarioTimeoutError(
            f"scenario exceeded {seconds:g}s timeout"
        ) from exc
    finally:
        executor.shutdown(wait=False)


class ScenarioResult(BaseModel):
    """Structured result for one executed scenario."""

    scenario_id: str
    category: str
    customer_message: str
    order_id: str | None = None
    agent_model: str | None = None
    answer: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    metrics: list[MetricScore] = Field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "ok" and all(
            check.status != "fail" for check in self.checks
        )


class CategorySummary(BaseModel):
    total: int = 0
    successful: int = 0
    failed: int = 0
    passed: int = 0


class EvaluationSummary(BaseModel):
    total: int = 0
    successful: int = 0
    failed: int = 0
    scenarios_passed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_error: int = 0
    metrics_passed: int = 0
    metrics_failed: int = 0
    metrics_error: int = 0
    per_category: dict[str, CategorySummary] = Field(default_factory=dict)


class EvaluationRunResult(BaseModel):
    label: str
    run_id: str
    results_path: str
    summary_path: str
    summary: EvaluationSummary
    results: list[ScenarioResult] = Field(default_factory=list)


ScenarioStartHook = Callable[[Scenario, int, int], None]
ScenarioResultHook = Callable[[ScenarioResult, int, int], None]


def summarize(results: Sequence[ScenarioResult]) -> EvaluationSummary:
    summary = EvaluationSummary(total=len(results))
    for result in results:
        category = summary.per_category.setdefault(result.category, CategorySummary())
        category.total += 1
        if result.status == "ok":
            summary.successful += 1
            category.successful += 1
        else:
            summary.failed += 1
            category.failed += 1
        if result.passed:
            summary.scenarios_passed += 1
            category.passed += 1
        for check in result.checks:
            if check.status == "pass":
                summary.checks_passed += 1
            elif check.status == "fail":
                summary.checks_failed += 1
            else:
                summary.checks_error += 1
        for metric in result.metrics:
            if metric.error is not None:
                summary.metrics_error += 1
            elif metric.success is True:
                summary.metrics_passed += 1
            elif metric.success is False:
                summary.metrics_failed += 1
            else:
                summary.metrics_error += 1
    return summary


class EvaluationRunner:
    """Runs the evaluation dataset against a freshly isolated agent per scenario."""

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = None,
        results_dir: str | Path | None = None,
        label: str = "baseline",
        reference_date: date | None = DEFAULT_REFERENCE_DATE,
        settings: AgentSettings | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        configure_tracing: bool = True,
        store_data: dict[str, Any] | None = None,
        scenario_timeout: float | None = DEFAULT_SCENARIO_TIMEOUT,
        use_metrics: bool = False,
        metrics: Sequence[tuple[str, Any]] | None = None,
        judge: Any | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.results_dir = (
            Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        )
        self.label = label
        self.reference_date = reference_date
        self.settings = settings
        self.model_factory = model_factory
        self.configure_tracing = configure_tracing
        self.scenario_timeout = scenario_timeout
        self.use_metrics = use_metrics
        self._metrics = list(metrics) if metrics is not None else None
        self._judge = judge
        if store_data is None:
            store_data = json.loads(DEFAULT_MOCK_STORE_PATH.read_text(encoding="utf-8"))
        self._store_data = store_data

    def _build_agent(self, scenario: Scenario) -> Any:
        model = self.model_factory() if self.model_factory is not None else None
        return create_baseline_agent(
            store=OrderStore(data=copy.deepcopy(self._store_data)),
            registry=EscalationRegistry(),
            reference_date=self.reference_date,
            settings=self.settings,
            model=model,
            configure_tracing=self.configure_tracing,
            metadata={"scenario_id": scenario.id, "scenario_category": scenario.category},
        )

    def _known_model(self) -> str | None:
        return self.settings.model if self.settings is not None else None

    def _resolve_judge(self) -> Any:
        if self._judge is None:
            from .judge import create_judge_model

            self._judge = create_judge_model()
        return self._judge

    def _resolve_metrics(self) -> list[tuple[str, Any]]:
        if self._metrics is None:
            from .metric_evaluation import build_metrics

            self._metrics = build_metrics(self._resolve_judge())
        return self._metrics

    def _evaluate_metrics(self, scenario: Scenario, run: Any) -> list[MetricScore]:
        if not self.use_metrics:
            return []
        return evaluate_scenario(scenario, run, self._resolve_metrics())

    def _run_scenario(self, scenario: Scenario) -> ScenarioResult:
        started_at = datetime.now(UTC)
        try:
            run = _run_with_timeout(
                self.scenario_timeout,
                lambda: self._build_agent(scenario).run(scenario.customer_message),
            )
        except Exception as exc:  # noqa: BLE001 - record any error and continue
            finished_at = datetime.now(UTC)
            return ScenarioResult(
                scenario_id=scenario.id,
                category=scenario.category,
                customer_message=scenario.customer_message,
                order_id=scenario.order_id,
                agent_model=self._known_model(),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=_elapsed_ms(started_at, finished_at),
            )

        finished_at = datetime.now(UTC)
        return ScenarioResult(
            scenario_id=scenario.id,
            category=scenario.category,
            customer_message=scenario.customer_message,
            order_id=scenario.order_id,
            agent_model=run.model,
            answer=run.answer,
            tool_calls=run.tool_calls,
            checks=run_checks(
                scenario, run, self._resolve_judge() if self.use_metrics else None
            ),
            metrics=self._evaluate_metrics(scenario, run),
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=_elapsed_ms(started_at, finished_at),
        )

    def _append_result(self, path: Path, result: ScenarioResult) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(result.model_dump_json() + "\n")

    def run(
        self,
        scenarios: Sequence[Scenario] | None = None,
        *,
        on_start: ScenarioStartHook | None = None,
        on_result: ScenarioResultHook | None = None,
    ) -> EvaluationRunResult:
        scenarios = (
            list(scenarios)
            if scenarios is not None
            else load_scenarios(self.dataset_path)
        )

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        safe_label = _sanitize_label(self.label)
        results_path = self.results_dir / f"{safe_label}_{run_id}.jsonl"
        summary_path = self.results_dir / f"{safe_label}_{run_id}.summary.json"

        total = len(scenarios)
        results: list[ScenarioResult] = []
        for index, scenario in enumerate(scenarios, start=1):
            if on_start is not None:
                on_start(scenario, index, total)
            result = self._run_scenario(scenario)
            results.append(result)
            self._append_result(results_path, result)
            if on_result is not None:
                on_result(result, index, total)

        summary = summarize(results)
        summary_path.write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )

        return EvaluationRunResult(
            label=self.label,
            run_id=run_id,
            results_path=str(results_path),
            summary_path=str(summary_path),
            summary=summary,
            results=results,
        )

    def resume(
        self,
        resume_path: str | Path,
        scenarios: Sequence[Scenario] | None = None,
        *,
        on_start: ScenarioStartHook | None = None,
        on_result: ScenarioResultHook | None = None,
    ) -> EvaluationRunResult:
        """Continue an interrupted run from its existing JSONL results file.

        Completed scenarios (status ``ok``) are skipped; scenarios that errored
        are retried. New results are appended to the same file, the existing
        summary file is refreshed, and the original label and run ID are kept.
        """
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume results file not found: {resume_path}")

        label, run_id = _parse_run_filename(resume_path)
        effective: dict[str, ScenarioResult] = {}
        for result in _read_results(resume_path):
            effective[result.scenario_id] = result

        scenarios = (
            list(scenarios)
            if scenarios is not None
            else load_scenarios(self.dataset_path)
        )
        completed = {
            scenario_id
            for scenario_id, result in effective.items()
            if result.status == "ok"
        }

        to_run = [scenario for scenario in scenarios if scenario.id not in completed]
        total = len(to_run)
        for index, scenario in enumerate(to_run, start=1):
            if on_start is not None:
                on_start(scenario, index, total)
            result = self._run_scenario(scenario)
            effective[scenario.id] = result
            self._append_result(resume_path, result)
            if on_result is not None:
                on_result(result, index, total)

        ordered: list[ScenarioResult] = []
        seen: set[str] = set()
        for scenario in scenarios:
            if scenario.id in effective and scenario.id not in seen:
                ordered.append(effective[scenario.id])
                seen.add(scenario.id)
        for scenario_id, result in effective.items():
            if scenario_id not in seen:
                ordered.append(result)

        summary = summarize(ordered)
        summary_path = resume_path.parent / f"{resume_path.stem}.summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        return EvaluationRunResult(
            label=label,
            run_id=run_id,
            results_path=str(resume_path),
            summary_path=str(summary_path),
            summary=summary,
            results=ordered,
        )

    def rerun_scenario(
        self,
        resume_path: str | Path,
        scenario_id: str,
        scenarios: Sequence[Scenario] | None = None,
        *,
        on_start: ScenarioStartHook | None = None,
        on_result: ScenarioResultHook | None = None,
    ) -> EvaluationRunResult:
        """Rerun one scenario and replace its record in an existing results file.

        The scenario must already exist in the run's JSONL file; this method never
        adds a new scenario to a run. The original label and run ID are preserved,
        the summary is recalculated from the complete updated result set, and the
        JSONL file is rewritten so no duplicate scenario records remain.
        """
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume results file not found: {resume_path}")

        label, run_id = _parse_run_filename(resume_path)
        existing = _read_results(resume_path)
        effective: dict[str, ScenarioResult] = {}
        order: list[str] = []
        for result in existing:
            if result.scenario_id not in effective:
                order.append(result.scenario_id)
            effective[result.scenario_id] = result

        scenarios = (
            list(scenarios)
            if scenarios is not None
            else load_scenarios(self.dataset_path)
        )
        scenario_by_id = {scenario.id: scenario for scenario in scenarios}
        if scenario_id not in scenario_by_id:
            raise ValueError(
                f"Scenario ID {scenario_id!r} is not part of the dataset."
            )
        if scenario_id not in effective:
            raise ValueError(
                f"Scenario ID {scenario_id!r} has no existing record in {resume_path}."
            )

        scenario = scenario_by_id[scenario_id]
        if on_start is not None:
            on_start(scenario, 1, 1)
        result = self._run_scenario(scenario)
        effective[scenario_id] = result
        if on_result is not None:
            on_result(result, 1, 1)

        ordered = [effective[scenario_key] for scenario_key in order]

        _write_results(resume_path, ordered)

        summary = summarize(ordered)
        summary_path = resume_path.parent / f"{resume_path.stem}.summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        return EvaluationRunResult(
            label=label,
            run_id=run_id,
            results_path=str(resume_path),
            summary_path=str(summary_path),
            summary=summary,
            results=ordered,
        )


def _print_summary(console: Console, run_result: EvaluationRunResult) -> None:
    summary = run_result.summary
    console.print(
        f"[bold]{run_result.label}[/bold] run {run_result.run_id}: "
        f"{summary.successful}/{summary.total} executed, "
        f"{summary.failed} errored, "
        f"{summary.scenarios_passed} passed behavior checks"
    )
    console.print(
        f"checks: {summary.checks_passed} passed, "
        f"{summary.checks_failed} failed, {summary.checks_error} error"
    )
    console.print(
        f"metrics: {summary.metrics_passed} passed, "
        f"{summary.metrics_failed} failed, {summary.metrics_error} error"
    )

    table = Table(title="Per category")
    table.add_column("Category")
    table.add_column("Total", justify="right")
    table.add_column("Executed", justify="right")
    table.add_column("Errored", justify="right")
    table.add_column("Passed", justify="right")
    for category, counts in sorted(summary.per_category.items()):
        table.add_row(
            category,
            str(counts.total),
            str(counts.successful),
            str(counts.failed),
            str(counts.passed),
        )
    console.print(table)
    console.print(f"results: {run_result.results_path}")
    console.print(f"summary: {run_result.summary_path}")


def _make_progress_hooks(console: Console):
    def on_start(scenario: Scenario, index: int, total: int) -> None:
        console.print(
            f"[cyan]\\[{index}/{total}][/cyan] running "
            f"[bold]{scenario.id}[/bold] ({scenario.category}) ...",
            highlight=False,
        )

    def on_result(result: ScenarioResult, index: int, total: int) -> None:
        if result.status == "ok":
            failed = sum(1 for check in result.checks if check.status == "fail")
            calls = len(result.tool_calls)
            verdict = (
                "[green]ok[/green]"
                if failed == 0
                else f"[yellow]{failed} check(s) failed[/yellow]"
            )
            console.print(f"    [dim]done:[/dim] {verdict}, {calls} tool call(s)")
            if result.metrics:
                parts = []
                for metric in result.metrics:
                    if metric.error is not None:
                        parts.append(f"{metric.name}=err")
                    else:
                        parts.append(f"{metric.name}={metric.score:.2f}")
                console.print("    [dim]metrics:[/dim] " + ", ".join(parts))
        else:
            console.print(f"    [red]error:[/red] {result.error}")

    return on_start, on_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentEval evaluation.")
    parser.add_argument("--label", default="baseline", help="Run label.")
    parser.add_argument("--dataset", help="Path to the evaluation dataset JSON.")
    parser.add_argument("--results-dir", help="Directory for result files.")
    parser.add_argument(
        "--reference-date",
        help="Fix the refund reference date as YYYY-MM-DD (default 2026-11-24).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N scenarios."
    )
    parser.add_argument(
        "--resume",
        help="Resume an interrupted run from an existing results JSONL file.",
    )
    parser.add_argument(
        "--scenario-id",
        help="Rerun one scenario from an existing --resume results file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_SCENARIO_TIMEOUT,
        help=f"Per-scenario timeout in seconds (default {DEFAULT_SCENARIO_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip DeepEval metrics and run deterministic checks only.",
    )
    args = parser.parse_args(argv)

    if args.scenario_id and not args.resume:
        parser.error("--scenario-id requires --resume")

    reference_date = (
        date.fromisoformat(args.reference_date)
        if args.reference_date
        else DEFAULT_REFERENCE_DATE
    )

    console = Console()
    use_metrics = not args.no_metrics
    metrics: list[tuple[str, Any]] | None = None
    judge: Any | None = None
    if use_metrics:
        from .judge import create_judge_model
        from .metric_evaluation import build_metrics

        try:
            judge = create_judge_model()
            metrics = build_metrics(judge)
        except Exception as exc:  # noqa: BLE001 - configuration errors are user-facing
            console.print(
                Panel(
                    f"{exc}\n\nSet OPENROUTER_API_KEY, OPENROUTER_BASE_URL and "
                    "JUDGE_MODEL, or run with --no-metrics.",
                    title="Judge configuration error",
                    border_style="red",
                )
            )
            return 1

    runner = EvaluationRunner(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        label=args.label,
        reference_date=reference_date,
        scenario_timeout=args.timeout,
        use_metrics=use_metrics,
        metrics=metrics,
        judge=judge,
    )
    scenarios = load_scenarios(args.dataset)
    if args.limit is not None and args.scenario_id is None:
        scenarios = scenarios[: args.limit]

    on_start, on_result = _make_progress_hooks(console)

    if args.scenario_id:
        console.print(
            f"[bold]Rerunning[/bold] {args.scenario_id} in {args.resume} "
            f"(timeout {args.timeout:g}s)"
        )
        try:
            run_result = runner.rerun_scenario(
                args.resume,
                args.scenario_id,
                scenarios,
                on_start=on_start,
                on_result=on_result,
            )
        except (FileNotFoundError, ValueError) as exc:
            console.print(
                Panel(str(exc), title="Scenario rerun error", border_style="red")
            )
            return 1
    elif args.resume:
        console.print(
            f"[bold]Resuming[/bold] {args.resume} "
            f"({len(scenarios)} scenario(s) considered, timeout {args.timeout:g}s)"
        )
        run_result = runner.resume(
            args.resume, scenarios, on_start=on_start, on_result=on_result
        )
    else:
        console.print(
            f"[bold]{args.label}[/bold]: running {len(scenarios)} scenario(s) "
            f"(timeout {args.timeout:g}s, metrics={'on' if use_metrics else 'off'})"
        )
        run_result = runner.run(scenarios, on_start=on_start, on_result=on_result)

    _print_summary(console, run_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
