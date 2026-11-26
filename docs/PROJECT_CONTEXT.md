# Project Context

**AgentEval — An LLM evaluation framework for testing AI agents using DeepEval, custom metrics, regression testing, and automated CI/CD pipelines.**

The agent under test is an AI Customer Support Agent for a fictional electronics store.

## The problem

Tool-using LLM agents can produce a fluent answer while still behaving incorrectly: calling the wrong tool, passing bad arguments, skipping required checks, or promising an outcome the business rules do not allow. A customer support agent is a useful demonstration because its correct behavior depends on real order data and explicit policies such as return windows and cancellation rules.

The repository now implements the deterministic foundation (a mock store, a validated data access layer, five customer support tools, and unit tests) and a Baseline Agent built with LangChain `create_agent` over a Google Gemini model, with LangSmith tracing and structured execution traces. The evaluation runner, judge, and comparison framework are planned next. See `docs/DEVELOPMENT_STATUS.md` for the verified state.

## What this project demonstrates

- Designing narrow, deterministic tools with explicit success and failure results.
- Separating mock data access from business logic.
- Modeling tool inputs and outputs with Pydantic.
- Building a fixed, behavior-focused evaluation dataset.
- Evaluating a tool-using agent beyond its final prose.
- Comparing a Baseline Agent with an Improved Agent under the same scenarios.
- Turning failures into regression checks and CI gates (planned).
- Investigating individual failures instead of reporting only aggregate scores.

The intended audience is AI Engineer / Applied AI Engineer recruiters and engineering teams. The project should read as an honest engineering exercise: measured outcomes, documented limitations, and reproducible commands, not inflated claims.

## Why final-answer evaluation is insufficient

A response can be relevant and well written while the agent:

- invents an order status instead of calling `get_order_status`;
- calls `cancel_order` on a shipped order;
- approves a return without checking the 30-day window or the item condition;
- skips escalation for a damaged or missing item;
- reports a refund or cancellation that never happened.

Evaluating only the final text misses these failures. The framework therefore needs to inspect the tool calls, their arguments, their results, the call order, and the final response's grounding in those results. Answer quality still matters, but it is one layer, not the whole evaluation.

## Goals

1. Provide a realistic, deterministic tool layer for an e-commerce support agent.
2. Build a Baseline Agent that uses those tools through a structured execution trace.
3. Evaluate answer quality, agent behavior, and reliability on a fixed dataset.
4. Improve the agent based on observed failures and re-run the same scenarios.
5. Add regression tests and CI so quality changes are visible before merge.
6. Document results, disagreements between judge metrics, and limitations.

Non-goals at the current stage: production customer support, real payments or refunds, multi-tenant services, and a general-purpose agent framework.

## Two systems, kept separate

| System | Responsibility | Current status |
| --- | --- | --- |
| Agent under test | Reads a customer message, chooses tools, supplies arguments, and writes the customer response. | Implemented (Baseline; not yet evaluated) |
| Evaluation framework | Runs fixed scenarios, records traces, applies deterministic checks and judge metrics, compares runs, and reports results. | Planned |

The tools belong to the application side, not the evaluation side. Evaluation code must not be embedded in tools, and tools must not know which scenario is running.

## Intended technology stack

| Area | Technology | Status |
| --- | --- | --- |
| Language | Python 3.13 | In use |
| Environment | uv, `.python-version`, `uv.lock` | In use |
| Data contracts | Pydantic v2 | In use |
| Tests | pytest | In use |
| Mock data | JSON (`data/mock_store.json`, 24 orders) | In use |
| Evaluation scenarios | JSON (`data/evaluation_dataset.json`, 14 scenarios) | In use |
| Agent | LangChain `create_agent` + Google Gemini (AI Studio) | Implemented (Baseline) |
| Tracing | LangSmith | Implemented |
| CLI | Rich | Implemented |
| Evaluation | DeepEval and custom metrics / judges | Implemented; judge uses a Groq model via OpenRouter |
| Regression and CI | Pytest-based quality gates, GitHub Actions | Partially implemented; CI runs lint and unit tests |

The Baseline Agent, its configuration, LangChain tool wrappers, structured traces, and the Rich CLI are implemented. The Google Gemini agent uses the direct Google AI Studio API; the judge uses a Groq model through OpenRouter with `langchain-openai`. DeepEval, the OpenRouter judge, the evaluation runner, and the metrics are implemented but have not been run against the real judge yet.

## Definition of success

The project is successful when it can:

- run the same 14 scenarios against Baseline and Improved agents;
- show which tool-selection, argument, sequence, escalation, and grounding failures occurred;
- report both deterministic results and LLM-judge results with their limitations;
- reproduce the comparison from a clean environment;
- fail a regression check when a known-good behavior breaks.

That comparison has not been run yet; no Baseline or Improved results exist.