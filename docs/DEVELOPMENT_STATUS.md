# Development Status

**As of:** 2026-11-25

## Current stage

The deterministic tool layer is complete and tested. The Baseline Agent is implemented with LangChain `create_agent`, Google Gemini, and LangSmith tracing, and produces structured execution traces. There is still no evaluation runner, DeepEval integration, experiment report, regression gate, or CI workflow.

**Partially implemented:** the evaluation dataset is present, but no runner consumes it yet, so it is not an implemented evaluation capability. The judge settings are prepared, but the judge itself does not exist.

## Completed components

| Component | Evidence |
| --- | --- |
| uv project | `pyproject.toml`, `uv.lock`, `.python-version` (3.13). |
| Runtime dependencies | Pydantic v2, plus `langchain`, `langchain-google-genai`, `langchain-openai`, `langsmith`, `rich`, `python-dotenv`. |
| Test dependency | pytest (`pytest>=9.2.0`) in the `dev` dependency group. |
| Mock data | `data/mock_store.json`: 22 orders, USD, statuses 4 processing / 3 shipped / 15 delivered. |
| Evaluation dataset | `data/evaluation_dataset.json`: 25 scenarios, 5 per category; 9 require escalation. |
| Data access | `OrderStore` loads via `pathlib`, validates Pydantic models, and mutates only in memory. |
| Tool models | `Order`, `OrderItem`, `StoreData`, and structured result models. |
| Five tools | `get_order_status`, `get_order_details`, `check_refund_eligibility`, `cancel_order`, `escalate_to_human`. |
| Escalation | In-memory `EscalationRegistry`; sequential process-local IDs; supports no order ID. |
| Configuration | `src/agent/config.py`: independent agent, judge, and LangSmith settings from `.env`; `SecretStr` secrets. |
| Tool wrappers | `src/agent/toolkit.py`: LangChain `@tool` wrappers delegating to the existing tools. |
| Structured traces | `src/agent/traces.py`: `ToolCallRecord`/`AgentRun`/`AgentEvent` and trace-extraction helpers. |
| Baseline Agent | `src/agent/baseline_agent.py`: `create_agent` graph over the tools with a Google Gemini model; `run` and streaming `stream`. |
| CLI | `src/agent/cli.py`: Rich title panel, thinking spinner, streamed tool activity, tool-call table, answer panel, graceful errors, interactive loop. |
| Unit tests | 38 tool tests in `tests/test_customer_support_tools.py`; 16 scaffolding tests in `tests/test_agent_scaffolding.py`; 8 orchestration tests in `tests/test_baseline_agent.py` (mocked models); 2 CLI tests in `tests/test_cli.py`. |
| Documentation | `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION_STRATEGY.md`, this file. |

## Verification

```text
$ uv run pytest -q
64 passed in 0.58s

$ uvx ruff check src tests
All checks passed!
```

Verified locally on 2026-11-25 with Python 3.13.18. Tests use temporary copies of the mock store, mocked models (no API calls), and an offline `conftest.py` that disables LangSmith tracing. The real data files are not modified. Manual smoke runs of the Baseline Agent and the Rich CLI against Gemini confirmed streamed tool activity, grounded answers, and trace export under the `baseline` tag; in one run the minimal baseline cancelled an order without first checking its status, an expected baseline weakness rather than an artificial one. These are not evaluations and are not part of the test suite. There is no CI workflow, so these results are local only.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Complete customer support tools and unit tests | **Completed** — five tools implemented; 38 tests pass. |
| 2 | Implement Baseline Agent | **Completed** — minimal-prompt `create_agent` + Gemini, LangSmith tracing with `baseline` tag and metadata, structured `AgentRun` traces, Rich CLI, 26 agent/CLI tests (mocked, no API calls). Agent behavior is not yet evaluated. |
| 3 | Implement the evaluation runner | **Planned** — next. |
| 4 | Implement the three evaluation layers | Planned. |
| 5 | Run and analyze Baseline evaluation | Planned. |
| 6 | Improve agent behavior based on observed failures | Planned. |
| 7 | Run Improved evaluation and compare results | Planned. |
| 8 | Add regression testing | Planned. |
| 9 | Integrate evaluation into GitHub Actions | Planned. |
| 10 | Document results and limitations | Planned. |

## What intentionally does not exist yet

- No evaluation runner consuming `evaluation_dataset.json`.
- No DeepEval or custom metric implementation.
- No judge model or judge invocation (only `JudgeSettings` configuration is prepared).
- No experiment runner, stored reports, or baseline/improved results.
- No regression thresholds or GitHub Actions workflow.
- No persistence for cancellation or escalation state.
- No multi-turn conversation state; the agent handles one message per run.

## Immediate next step

Start Milestone 3: implement the evaluation runner that loads `evaluation_dataset.json`, executes each scenario against the Baseline Agent with a fresh `OrderStore` and `EscalationRegistry` and a fixed `reference_date`, captures the structured `AgentRun` trace, and persists results. Do not add DeepEval or judge metrics in that step; deterministic behavior checks come first.

## Documentation maintenance

Update this file whenever a milestone changes state, a dependency is added, or verification results change. Update `docs/ARCHITECTURE.md` when component boundaries or data flow change, and `docs/EVALUATION_STRATEGY.md` when metric definitions or comparison rules change.