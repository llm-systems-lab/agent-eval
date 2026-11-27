# Development Status

**As of:** 2026-11-27

## Current stage

The deterministic tool layer is complete and tested. The Baseline Agent is implemented with LangChain `create_agent`, Google Gemini, and LangSmith tracing, and produces structured execution traces. The evaluation runner executes the dataset sequentially with an isolated store per scenario, applies deterministic behavior checks and LLM-judge checks, and saves results incrementally with resume and single-scenario rerun. DeepEval is integrated: six metrics (Tool Correctness, Argument Correctness, Task Completion, Step Efficiency, Tool Outcome Consistency, Business Policy Compliance) run per scenario through an OpenRouter-backed judge and are stored in the JSONL results. `src/agent/regression.py` (`uv run eval-regression`) compares a candidate run against a baseline. CI (`.github/workflows/ci.yml`) runs Ruff and the offline unit tests on every push and on pull requests to `main`; `.github/workflows/llm-evaluation.yml` runs the live evaluation manually and fails on regressions. Baseline and improved experiments have been run and stored under `results/`. There is still no standalone experiment report and no per-scenario threshold gates.

**Partially implemented:** the baseline and improved runs used different revisions of `data/evaluation_dataset.json`, so the comparison is not perfectly controlled (see `docs/EVALUATION_STRATEGY.md`). `expected_facts` and `forbidden_behaviors` are scored by LLM-judge checks (`judge_checks`) using the shared OpenRouter judge, in addition to the deterministic checks for tool calls, arguments, escalation, and the cancellation precondition; these judge checks have now run against the real judge and occasionally return infrastructure errors (structured-output and length-limit failures), recorded as `error`.

## Completed components

| Component | Evidence |
| --- | --- |
| uv project | `pyproject.toml`, `uv.lock`, `.python-version` (3.13). |
| Runtime dependencies | Pydantic v2, plus `langchain`, `langchain-google-genai`, `langchain-openai`, `langsmith`, `rich` (pinned `<15` for DeepEval), `python-dotenv`, `deepeval`. |
| Test dependency | pytest (`pytest>=9.2.0`) in the `dev` dependency group. |
| Mock data | `data/mock_store.json`: 24 orders, USD, statuses 4 processing / 3 shipped / 17 delivered. |
| Evaluation dataset | `data/evaluation_dataset.json`: 14 scenarios across five categories; 6 require escalation. |
| Data access | `OrderStore` loads via `pathlib`, validates Pydantic models, and mutates only in memory. |
| Tool models | `Order`, `OrderItem`, `StoreData`, and structured result models. |
| Five tools | `get_order_status`, `get_order_details`, `check_refund_eligibility`, `cancel_order`, `escalate_to_human`. |
| Escalation | In-memory `EscalationRegistry`; sequential process-local IDs; supports no order ID. |
| Configuration | `src/agent/config.py`: independent agent, judge, and LangSmith settings from `.env`; `SecretStr` secrets. |
| Tool wrappers | `src/agent/toolkit.py`: LangChain `@tool` wrappers delegating to the existing tools. |
| Structured traces | `src/agent/traces.py`: `ToolCallRecord`/`AgentRun`/`AgentEvent` and trace-extraction helpers. |
| Baseline Agent | `src/agent/baseline_agent.py`: `create_agent` graph over the tools with a Google Gemini model; `run` and streaming `stream`. |
| CLI | `src/agent/cli.py`: Rich title panel, thinking spinner, streamed tool activity, tool-call table, answer panel, graceful errors, interactive loop. |
| Scenario loader | `src/agent/scenarios.py`: read-only loader for `data/evaluation_dataset.json`. |
| Behavior checks | `src/agent/behavior_checks.py`: deterministic checks for expected tool calls, arguments, escalation, and the cancellation precondition; `expected_facts`/`forbidden_behaviors` handled by `judge_checks`. |
| Evaluation runner | `src/agent/evaluation_runner.py`: sequential execution with per-scenario isolation, incremental JSONL saving, resume from an existing results file, single-scenario rerun that replaces its record, per-scenario timeout, live progress output, DeepEval metrics, and summary. |
| Judge model | `src/agent/judge.py`: `OpenRouterJudge` (`DeepEvalBaseLLM`) via `langchain-openai` on OpenRouter, from `JudgeSettings`. |
| DeepEval metrics | `src/agent/metric_evaluation.py`: Tool Correctness, Argument Correctness (policy-aware prompt), Task Completion (policy-aware prompt), Step Efficiency (calibrated prompt); `src/agent/custom_metrics.py`: Tool Outcome Consistency and Business Policy Compliance (both `GEval`); scores/errors stored per scenario. |
| Judge checks | `src/agent/judge_checks.py`: LLM-judge checks for `expected_facts` and `forbidden_behaviors` via the shared OpenRouter judge; `pass`/`fail`/`error`. |
| Unit tests | 38 tool tests (`tests/test_customer_support_tools.py`); 29 scaffolding (`tests/test_agent_scaffolding.py`); 8 orchestration (`tests/test_baseline_agent.py`); 2 CLI (`tests/test_cli.py`); 39 evaluation (`tests/test_evaluation_runner.py`); 17 metric (`tests/test_metric_evaluation.py`); 31 custom metric (`tests/test_custom_metrics.py`); 11 judge-check (`tests/test_judge_checks.py`); 16 regression (`tests/test_regression.py`); 191 tests total. |
| Documentation | `README.md`, `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION_STRATEGY.md`, this file. |
| Baseline experiment | `results/baseline_20261126T132600_cf9726.jsonl`/`.summary.json`: 14 scenarios; 83 checks passed, 1 failed; 72 metrics passed, 12 failed. |
| Improved experiment | `results/improvements_20261127T082918_e85922.jsonl`/`.summary.json`: 14 scenarios; 75 checks passed, 8 failed, 1 check error; 80 metrics passed, 4 failed, 0 metric errors. |
| Regression comparison | `src/agent/regression.py` (`uv run eval-regression`): baseline-vs-candidate comparison of scenarios, checks, and metrics with a configurable score tolerance; metric errors reported separately. |

## Verification

```text
$ uv run pytest -q
191 passed in 6.41s

$ uvx ruff check src tests
All checks passed!
```

Verified locally on 2026-11-27 with Python 3.13. Tests use temporary copies of the mock store, mocked models and fake metrics (no API calls), and an offline `conftest.py` that disables LangSmith tracing; the real data files are not modified. Manual smoke runs of the Baseline Agent and the Rich CLI against Gemini confirmed streamed tool activity, grounded answers, and trace export under the `baseline` tag. The baseline experiment (14 scenarios) and an improved experiment (14 scenarios) have been run against Gemini with the DeepEval metrics and the OpenRouter judge, and their results are stored under `results/`. A targeted 3-scenario run after the latest policy changes passed 3/3 scenarios, 18/18 checks, and 18/18 metrics. Running the live evaluation requires `GOOGLE_API_KEY` and, for metrics, `OPENROUTER_API_KEY`; CI runs Ruff and pytest offline. The baseline and improved runs used different dataset revisions, so the comparison is not perfectly controlled. The offline test/lint results above are the authoritative local verification.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Complete customer support tools and unit tests | **Completed** — five tools implemented; 38 tests pass. |
| 2 | Implement Baseline Agent | **Completed** — `create_agent` + Gemini, LangSmith tracing with `baseline` tag and metadata, structured `AgentRun` traces, Rich CLI, 39 agent/CLI tests (mocked, no API calls). |
| 3 | Implement the evaluation runner | **Completed** — loader, deterministic checks, sequential runner with per-scenario isolation, incremental saving, resume, single-scenario rerun, per-scenario timeout, progress, summary, metrics integration, 39 tests. Has since run against Gemini (milestone 5). |
| 4 | Implement the three evaluation layers | **Implemented** — Layer 2 deterministic checks, Layer 1 DeepEval metrics (Tool Correctness, Argument Correctness, Task Completion, Step Efficiency, Tool Outcome Consistency, Business Policy Compliance), and Layer 1 judge checks for expected facts/forbidden behaviors, all via an OpenRouter judge that has now been run against live results. Layer 3 reliability (regression comparison) is implemented in milestones 8 and 9. |
| 5 | Run and analyze Baseline evaluation | **Completed** — the baseline experiment ran all 14 scenarios against Gemini with metrics; results in `results/baseline_20261126T132600_cf9726.*` (83 checks passed / 1 failed; 72 metrics passed / 12 failed). |
| 6 | Improve agent behavior based on observed failures | **In progress** — an improved prompt/policy revision was evaluated as a labeled run; a standalone experiment report is still pending. |
| 7 | Run Improved evaluation and compare results | **Completed** — an improved experiment ran all 14 scenarios; results in `results/improvements_20261127T082918_e85922.*` (75 checks passed / 8 failed / 1 check error; 80 metrics passed / 4 failed / 0 metric errors). Comparison caveat: the dataset differed between the baseline and improved runs. |
| 8 | Add regression testing | **Completed** — `src/agent/regression.py` compares a candidate run against a baseline by scenario, check, and metric, with a configurable score tolerance; metric errors are reported separately and have been compared against live results. |
| 9 | Integrate evaluation into GitHub Actions | **Completed** — `.github/workflows/ci.yml` runs Ruff and pytest on push and PRs to `main`; `.github/workflows/llm-evaluation.yml` runs the live evaluation manually and fails on regressions. |
| 10 | Document results and limitations | **In progress** — `README.md`, this file, and the strategy/architecture docs record the current state and limitations; a dedicated experiment report is still pending. |

## What intentionally does not exist yet

- No per-scenario threshold gates yet; the custom metrics report scores but do not fail the run.
- No standalone experiment report; the regression gate compares against a baseline only when the manual LLM-evaluation workflow is run.
- The baseline and improved runs used different revisions of `data/evaluation_dataset.json`, so the comparison is not perfectly controlled.
- No committed/frozen baseline by default; `results/` currently holds the baseline and the latest improved run.
- DeepEval metric scores are produced against the real judge, but vary with model, prompt, and sampling, and occasionally error (structured-output and length-limit failures).
- No persistence for cancellation or escalation state.
- No multi-turn conversation state; the agent handles one message per run.

## Immediate next step

Produce the experiment report. The baseline and improved experiments are complete; the next step is to explain the per-scenario failures and metric deltas, add per-scenario thresholds, and re-run baseline and improved on an identical dataset revision so the comparison is fully controlled. Use `--limit` to control API usage and `--no-metrics` for a deterministic-only pass.

## Documentation maintenance

Update this file whenever a milestone changes state, a dependency is added, or verification results change. Update `docs/ARCHITECTURE.md` when component boundaries or data flow change, and `docs/EVALUATION_STRATEGY.md` when metric definitions or comparison rules change.