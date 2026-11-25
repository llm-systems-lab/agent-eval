# AGENTS.md

Persistent instructions for Kilo Code sessions in this repository.

## Project

AgentEval builds and evaluates a tool-using AI Customer Support Agent for a fictional electronics store. The end goal is a repeatable evaluation framework: a Baseline Agent, an Improved Agent, DeepEval-based metrics, regression checks, and CI.

**Current state:** only the deterministic tool layer, mock data, evaluation dataset, unit tests, and uv project setup exist. There is no LLM agent, evaluation runner, DeepEval integration, experiment report, or CI workflow yet. Read `docs/DEVELOPMENT_STATUS.md` before starting work and `docs/ARCHITECTURE.md` before changing structure.

## Hard boundaries

- Do not modify `data/mock_store.json` or `data/evaluation_dataset.json` unless the user explicitly asks.
- Do not hardcode scenario IDs, expected tool calls, or expected outcomes in tools or agent prompts.
- Do not artificially manipulate, filter, or rerun away failing evaluation results.
- Do not change business rules (return window, cancellation status, escalation policy) without explicit approval.
- Do not add a database, web framework, agent framework, LLM provider, or DeepEval until the corresponding milestone is explicitly started.
- Keep tools deterministic and free of LLM calls, network calls, and wall-clock-dependent behavior where avoidable.
- Keep agent execution separate from evaluation logic, and terminal presentation separate from agent logic.
- Preserve structured execution traces once they exist; never replace them with free-text logs.
- Do not weaken or delete tests to make a change pass.

## Stack and layout

- Python 3.13 (pinned in `.python-version`), managed with `uv`.
- Pydantic v2 for data and tool contracts; pytest for tests.
- Runtime dependency: `pydantic`. Dev dependency: `pytest`.
- `src/` layout; pytest uses `pythonpath = ["src"]` from `pyproject.toml`.

```
data/                         mock store (22 orders) and fixed evaluation dataset (25 scenarios)
src/agent/tools/models.py     Pydantic data and result models
src/agent/tools/store.py      mock data loading and in-memory OrderStore
src/agent/tools/customer_support_tools.py
                              five tool functions and EscalationRegistry
tests/test_customer_support_tools.py
                              38 unit tests for the tool layer
docs/                         project, architecture, evaluation, and status documentation
```

## Existing tool interfaces

| Function | Contract |
| --- | --- |
| `get_order_status(order_id)` | Returns current status; explicit not-found result. |
| `get_order_details(order_id)` | Returns items, quantities, unit prices, order date, delivery date, total, currency. |
| `check_refund_eligibility(order_id, *, reference_date=None)` | Checks 30-day window, unused condition, original packaging; damaged/missing/tampered/unverifiable cases set `requires_human_support`; never processes a refund. |
| `cancel_order(order_id)` | Cancels only `processing` orders; updates the in-memory store to `cancelled` on success. |
| `escalate_to_human(order_id, reason)` | Records an in-memory escalation; accepts `order_id=None`; returns a process-local escalation ID. |

All five accept optional dependency injection (`store=`, or `registry=` for escalation) used by tests. Preserve these interfaces unless a task explicitly changes them.

## Commands

```bash
uv sync                 # install/refresh the locked environment
uv run pytest -q        # required verification for every code change
uv run pytest -q tests/test_customer_support_tools.py
uvx ruff check src tests  # optional lint check; no committed lint config yet
```

## Python standards

- Use type hints on public functions and non-trivial internals.
- Keep `from __future__ import annotations` at module start.
- Use Pydantic models for structured inputs/outputs instead of loose dictionaries.
- Use `pathlib.Path`; never depend on the current working directory for repository files.
- Keep data access (`store.py`) separate from business rules (`customer_support_tools.py`).
- Prefer small, explicit functions and deterministic messages over clever abstractions.
- Do not add comments unless they explain non-obvious constraints.

## Testing requirements

- Add or update tests for every behavior change, including invalid inputs, unknown orders, and boundary conditions.
- Use isolated in-memory stores or temporary copies of `data/mock_store.json`; tests must never mutate the real data files.
- Pass a fixed `reference_date` when testing refund eligibility.
- Use a fresh `EscalationRegistry` in escalation tests.
- Run `uv run pytest -q` before reporting completion and report the actual result.

## Workflow

1. Inspect the existing implementation and tests before editing; do not assume planned components exist.
2. Reuse existing models, store methods, tools, and helpers; avoid duplicate implementations.
3. Make the smallest change that satisfies the task.
4. Verify with tests (and lint when available).
5. Update `docs/DEVELOPMENT_STATUS.md` when implementation status changes, and the relevant architecture or strategy document when a design decision changes.

## Relevant documentation

- `docs/PROJECT_CONTEXT.md` — purpose, audience, and goals.
- `docs/ARCHITECTURE.md` — current and planned components.
- `docs/EVALUATION_STRATEGY.md` — three evaluation layers and methodology.
- `docs/DEVELOPMENT_STATUS.md` — verified status and milestone tracking.