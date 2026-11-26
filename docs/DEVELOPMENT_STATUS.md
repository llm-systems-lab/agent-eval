# Development Status

**As of:** 2026-11-25

## Current stage

The deterministic tool layer is complete and tested. The Baseline Agent is implemented with LangChain `create_agent`, Google Gemini, and LangSmith tracing, and produces structured execution traces. The evaluation runner executes the dataset sequentially with an isolated store per scenario, applies deterministic behavior checks, and saves results incrementally. DeepEval is integrated: six metrics (Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Tool Outcome Consistency, Business Policy Compliance) run per scenario through an OpenRouter-backed judge and are stored in the JSONL results. There is still no experiment report, regression gate, or CI workflow, and the baseline experiment has not been run yet.

**Partially implemented:** `expected_facts` and `forbidden_behaviors` are scored by LLM-judge checks (`judge_checks`) using the shared OpenRouter judge, in addition to the deterministic checks for tool calls, arguments, and escalation. Judge-based checks have not been run against the real judge yet.

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
| Behavior checks | `src/agent/behavior_checks.py`: deterministic checks for tool calls, arguments, and escalation; facts/forbidden handled by the DeepEval metrics. |
| Evaluation runner | `src/agent/evaluation_runner.py`: sequential execution with per-scenario isolation, incremental JSONL saving, resume from an existing results file, single-scenario rerun that replaces its record, per-scenario timeout, live progress output, DeepEval metrics, and summary. |
| Judge model | `src/agent/judge.py`: `OpenRouterJudge` (`DeepEvalBaseLLM`) via `langchain-openai` on OpenRouter, from `JudgeSettings`. |
| DeepEval metrics | `src/agent/metric_evaluation.py`: Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall; `src/agent/custom_metrics.py`: Tool Outcome Consistency and Business Policy Compliance; scores/errors stored per scenario. |
| Judge checks | `src/agent/judge_checks.py`: LLM-judge checks for `expected_facts` and `forbidden_behaviors` via the shared OpenRouter judge; `pass`/`fail`/`error`. |
| Unit tests | 38 tool tests (`tests/test_customer_support_tools.py`); 20 scaffolding (`tests/test_agent_scaffolding.py`); 8 orchestration (`tests/test_baseline_agent.py`); 2 CLI (`tests/test_cli.py`); 38 evaluation (`tests/test_evaluation_runner.py`); 10 metric (`tests/test_metric_evaluation.py`); 14 custom metric (`tests/test_custom_metrics.py`); 11 judge-check (`tests/test_judge_checks.py`). |
| Documentation | `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION_STRATEGY.md`, this file. |

## Verification

```text
$ uv run pytest -q
116 passed in 10.43s

$ uvx ruff check src tests
All checks passed!
```

Verified locally on 2026-11-25 with Python 3.13.18. Tests use temporary copies of the mock store, mocked models and fake metrics (no API calls), and an offline `conftest.py` that disables LangSmith tracing. The real data files are not modified. The runner's plumbing was smoke-tested with `uv run eval-runner --limit 0 --no-metrics` (no scenarios, no API calls). Manual smoke runs of the Baseline Agent and the Rich CLI against Gemini confirmed streamed tool activity, grounded answers, and trace export under the `baseline` tag; in one run the minimal baseline cancelled an order without first checking its status, an expected baseline weakness rather than an artificial one. The 14-scenario baseline experiment and the DeepEval metrics have not been run. There is no CI workflow, so these results are local only.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Complete customer support tools and unit tests | **Completed** — five tools implemented; 38 tests pass. |
| 2 | Implement Baseline Agent | **Completed** — minimal-prompt `create_agent` + Gemini, LangSmith tracing with `baseline` tag and metadata, structured `AgentRun` traces, Rich CLI, 26 agent/CLI tests (mocked, no API calls). Agent behavior is not yet evaluated. |
| 3 | Implement the evaluation runner | **Completed** — loader, deterministic checks, sequential runner with per-scenario isolation, incremental saving, resume, single-scenario rerun, per-scenario timeout, progress, summary, metrics integration, 38 tests. Not yet run against Gemini. |
| 4 | Implement the three evaluation layers | **In progress** — Layer 2 deterministic checks, Layer 1 DeepEval metrics (Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Tool Outcome Consistency, Business Policy Compliance), and Layer 1 judge checks for expected facts/forbidden behaviors are implemented via an OpenRouter judge; the judge has not been run yet. Layer 3 reliability is not started. |
| 5 | Run and analyze Baseline evaluation | **Next** — awaits approval to run the 14 scenarios against Gemini. |
| 6 | Improve agent behavior based on observed failures | Planned. |
| 7 | Run Improved evaluation and compare results | Planned. |
| 8 | Add regression testing | Planned. |
| 9 | Integrate evaluation into GitHub Actions | Planned. |
| 10 | Document results and limitations | Planned. |

## What intentionally does not exist yet

- No per-scenario threshold gates yet; the custom metrics report scores but do not fail the run.
- No experiment report, regression thresholds, or GitHub Actions workflow.
- No baseline/improved results yet (the baseline experiment has not been run).
- DeepEval metric scores have not been produced against the real judge.
- No persistence for cancellation or escalation state.
- No multi-turn conversation state; the agent handles one message per run.

## Immediate next step

Milestone 5: run the baseline experiment across all 14 scenarios with `uv run eval-runner --label baseline` (awaiting explicit approval before running against Gemini and the OpenRouter judge). Use `--limit` to control API usage and `--no-metrics` for a deterministic-only pass. Then analyze the deterministic checks, the DeepEval metric scores, and the per-category summary.

## Documentation maintenance

Update this file whenever a milestone changes state, a dependency is added, or verification results change. Update `docs/ARCHITECTURE.md` when component boundaries or data flow change, and `docs/EVALUATION_STRATEGY.md` when metric definitions or comparison rules change.