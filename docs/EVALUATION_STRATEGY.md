# Evaluation Strategy

## Status

The Baseline Agent, the evaluation runner, deterministic behavior checks, and the DeepEval metric layer are implemented. There is no experiment report, regression gate, or CI job yet, and no experiment has been run. This document defines how the three layers are implemented and evaluated.

## The fixed evaluation dataset

`data/evaluation_dataset.json` contains 14 scenarios across five categories: `order_status`, `order_details`, `refund`, `cancellation`, and `escalation`. Six scenarios set `should_escalate: true`.

Each scenario provides:

- `id` and `category`;
- `customer_message` and optional `order_id`;
- `expected_tool_calls` (the expected tool names, in order);
- `should_escalate`;
- `expected_facts` (qualitative requirements for the final answer);
- `forbidden_behaviors` (qualitative failure modes).

The same dataset must be used for the Baseline and Improved agents. Changing scenarios between runs makes a comparison meaningless. The dataset is read-only unless a deliberate dataset change is approved and its effect on prior results is documented.

## Layer 1 — Answer Quality

**Purpose:** judge the customer-facing response independently of the tool trace.

**Status:** implemented. Six DeepEval metrics — Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Tool Outcome Consistency, and Business Policy Compliance — run per scenario with an OpenRouter-backed Groq judge (`src/agent/judge.py`, `src/agent/metric_evaluation.py`, `src/agent/custom_metrics.py`). The custom metrics check tool-outcome consistency and business-policy compliance. Scores and errors are stored in the JSONL results. The metrics have not been run against the real judge yet. Potential metrics:

| Metric | Question |
| --- | --- |
| Answer relevancy | Does the answer address the customer's actual request? |
| Faithfulness / groundedness | Is the answer supported by tool results and known policy? |
| Hallucination detection | Did the answer invent statuses, amounts, dates, or outcomes? |
| Fact compliance | Are `expected_facts` present and `forbidden_behaviors` absent? |

Answer quality alone is not sufficient. A polite, relevant answer can still claim a cancellation succeeded when the tool rejected it. Layer 1 results must be read alongside behavior results.

## Layer 2 — Agent Behavior

**Purpose:** determine whether the agent used the tools correctly.

**Status:** implemented as deterministic checks over structured execution traces (`src/agent/behavior_checks.py`).

| Metric | What it checks |
| --- | --- |
| Tool selection correctness | Required tools were called; forbidden or unnecessary tools were avoided. |
| Tool argument correctness | Order IDs and other arguments match the scenario and tool contracts. |
| Tool execution correctness | The agent interpreted success, failure, and not-found results correctly. |
| Tool sequence correctness | Calls occur in the required order, such as checking status before cancellation. |
| Escalation correctness | `should_escalate` matches the actual escalation behavior, including no invented escalation IDs. |

These checks should be deterministic where the dataset permits. For example, cancellation scenarios expect `get_order_status` first and `cancel_order` only for processing orders. The dataset's `expected_tool_calls` list is a required-call list, so checks should distinguish "required call made" from "extra call made" rather than forcing a brittle exact string match without investigating the scenario. Missing-item and damaged-item scenarios require escalation instead of a self-service refund.

Behavior checks must read the trace, not infer tool calls from the final answer. A response claiming a refund or cancellation without a successful corresponding tool result is a failure.

## Layer 3 — Reliability

**Purpose:** show that quality is stable across changes, not just good in one run.

**Status:** planned — reliability reporting and regression gates come after the first baseline run.

Mechanisms:

- **Baseline vs Improved:** run the same 14 scenarios with both agents and compare layer scores, failure counts, and per-category behavior. Baseline and Improved must differ in documented, intentional ways; the comparison must not hide prompt or tool changes.
- **Regression testing:** convert known-good behaviors and previously observed failures into repeatable checks so they cannot silently regress.
- **Quality thresholds:** define minimum deterministic pass rates and judge-score thresholds before claiming an improvement.
- **CI/CD:** run unit tests, deterministic behavior checks, and the agreed evaluation thresholds in GitHub Actions. Judge-based checks may require API credentials and must degrade explicitly rather than silently skipping.

Reliability reporting should include run configuration, model identifiers when applicable, date, and any skipped or failed checks.

## Deterministic checks vs LLM-as-a-judge

| Aspect | Deterministic checks | LLM-as-a-judge |
| --- | --- | --- |
| Input | Dataset plus structured trace | Scenario, trace, final answer, rubric |
| Output | Pass/fail, counts, exact mismatches | Score and explanation |
| Strength | Reproducible, cheap, precise for tool behavior | Handles nuanced language and groundedness |
| Weakness | Brittle if scenario expectations are over-specified | Sensitive to prompt, model, and sampling |
| Use here | Tool calls, arguments, sequence, result interpretation, escalation | Relevancy, faithfulness, hallucination, fact compliance |

Both are needed. Deterministic checks provide a hard floor for behavior; judge metrics assess qualities that exact matching cannot capture. Judge scores are evidence, not ground truth, and should be reported with their limitations.

## Failure investigation

Aggregate scores are not enough. For every failing scenario, inspect:

1. the customer message and the run configuration;
2. the ordered tool calls and their arguments;
3. the tool results and whether the agent used them;
4. the final response's claims against those results and policy;
5. whether the failure is answer quality, tool behavior, escalation, or an infrastructure error.

Per-category and per-scenario results should be reported. A high aggregate score can hide a dangerous failure such as approving a refund for a damaged item or claiming a cancellation that never occurred.

## Comparing Baseline and Improved

A valid comparison requires:

- identical 14 scenarios and corresponding expected behavior;
- the same tool layer and mock data;
- recorded configuration for each run;
- both aggregate and per-scenario deltas;
- investigation of regressions, not only improvements;
- no removal, filtering, or rerunning of failing cases to improve a headline score.

"Improved" means a documented change produced a defensible improvement, not simply a higher average.

## Metric limitations

- The dataset is small (14 scenarios) and static; it demonstrates methodology rather than statistical confidence.
- Mock orders and fixed policies are simpler than production support.
- Exact tool-call matching can reward dogmatic behavior and miss valid alternative paths; this must be reviewed when defining checks.
- LLM judges vary with model version, prompt, temperature, and sampling. Scores may not reproduce exactly.
- Judge explanations can be plausible but wrong; disagreements with deterministic evidence must be resolved in favor of the trace for factual tool behavior.
- Refund checks depend on a reference date; evaluations must fix it so runs are comparable.
- Passing the dataset does not prove production safety, compliance, or privacy handling.

Results and limitations will be documented with the experiment report once the runner has been executed. No experiment results are claimed today.