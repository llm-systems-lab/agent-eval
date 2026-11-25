# Development Status

**As of:** 2026-11-24

## Current stage

The deterministic tool layer is complete and tested. The Baseline Agent has not been started. There is no evaluation runner, DeepEval integration, experiment report, regression gate, or CI workflow.

**Partially implemented:** none. The evaluation dataset is present, but no runner consumes it yet, so it is not an implemented evaluation capability.

## Completed components

| Component | Evidence |
| --- | --- |
| uv project | `pyproject.toml`, `uv.lock`, `.python-version` (3.13). |
| Runtime dependency | Pydantic v2 (`pydantic>=2.14.4`); only runtime dependency. |
| Test dependency | pytest (`pytest>=9.2.0`) in the `dev` dependency group. |
| Mock data | `data/mock_store.json`: 22 orders, USD, statuses 4 processing / 3 shipped / 15 delivered. |
| Evaluation dataset | `data/evaluation_dataset.json`: 25 scenarios, 5 per category; 9 require escalation. |
| Data access | `OrderStore` loads via `pathlib`, validates Pydantic models, and mutates only in memory. |
| Tool models | `Order`, `OrderItem`, `StoreData`, and structured result models. |
| Five tools | `get_order_status`, `get_order_details`, `check_refund_eligibility`, `cancel_order`, `escalate_to_human`. |
| Escalation | In-memory `EscalationRegistry`; sequential process-local IDs; supports no order ID. |
| Unit tests | 38 tests in `tests/test_customer_support_tools.py`. |
| Documentation | `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION_STRATEGY.md`, this file. |

## Verification

```text
$ uv run pytest -q
38 passed in 0.23s
```

Verified locally on 2026-11-24 with Python 3.13.18 and Pydantic 2.14.4. Tests use temporary copies of the mock store, and the real data files are not modified. There is no CI workflow, so this result is local only. No evaluation of an agent has been run.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Complete customer support tools and unit tests | **Completed** — five tools implemented; 38 tests pass. |
| 2 | Implement Baseline Agent | **Planned** — next. |
| 3 | Implement the evaluation runner | Planned. |
| 4 | Implement the three evaluation layers | Planned. |
| 5 | Run and analyze Baseline evaluation | Planned. |
| 6 | Improve agent behavior based on observed failures | Planned. |
| 7 | Run Improved evaluation and compare results | Planned. |
| 8 | Add regression testing | Planned. |
| 9 | Integrate evaluation into GitHub Actions | Planned. |
| 10 | Document results and limitations | Planned. |

## What intentionally does not exist yet

- No agent orchestration, prompt, or LLM provider.
- No structured execution trace implementation.
- No DeepEval or custom metric implementation.
- No experiment runner, stored reports, or baseline/improved results.
- No regression thresholds or GitHub Actions workflow.
- No persistence for cancellation or escalation state.

## Immediate next step

Start Milestone 2: implement the Baseline Agent using the existing five tools, and define the structured execution trace contract needed by the later evaluation runner. Do not add DeepEval or evaluation metrics in that step.

## Documentation maintenance

Update this file whenever a milestone changes state, a dependency is added, or verification results change. Update `docs/ARCHITECTURE.md` when component boundaries or data flow change, and `docs/EVALUATION_STRATEGY.md` when metric definitions or comparison rules change.