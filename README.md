# AI-Powered Suspicious Activity Detection Agent

An AML (Anti-Money Laundering) suspicious activity detection agent for retail banking. It accepts natural-language queries, routes them to the relevant analysis logic based on intent, and returns risk-scored findings with a short explanation and escalation recommendation.

## Problem Statement

Financial institutions run rule-based AML compliance systems that generate excessive false positives, overwhelming compliance teams, while techniques like structuring and smurfing evade conventional detection. This project builds a query-driven agent that performs targeted analysis — not a fixed pipeline — parsing user intent to invoke only the tools relevant to that specific query (e.g. a structuring query triggers pattern detection; a single-customer query does a direct lookup rather than broad dataset exploration).

The current implementation keeps that routing explicit in code so each query only runs the analysis path it needs. That makes the behavior easier to test, easier to explain, and closer to how a real triage workflow should behave than a one-size-fits-all batch pipeline.

## Dataset

- **Source**: [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), Kaggle.
- **Nature**: Fully synthetic transaction data simulating mobile money transfers, generated from aggregated real financial logs by the original researchers (not real customer data — no PII, compliant with the "no proprietary/confidential data" rule).
- **Schema**: `step` (time unit, 1 hour), `type` (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER), `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- **Raw file**: `Dataset/paysim dataset.csv`.
- **Working subset**: `paysim_working_subset.csv` is generated locally from the raw dataset by `build_working_subset()` and keeps all fraud rows plus a capped non-fraud sample for faster iteration.

The repository does not include the Kaggle raw CSV in Git. If the file is missing in your local checkout, download it from Kaggle and place it at `Dataset/paysim dataset.csv` before running the agent.

**Citation**: Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *28th European Modeling and Simulation Symposium (EMSS)*.

## Solution Approach

1. **Intent parsing**: The query is normalized and classified into one of five intents: structuring, customer risk, transaction risk, EDA/overview, or general.
2. **Dynamic routing**: The agent chooses a different analysis path depending on that intent. Structuring queries run band-based pattern detection, customer queries (once a valid entity ID is resolved) run direct entity lookups, transaction-risk queries run feature engineering followed by composite behavioral scoring, and EDA/overview queries return a dataset-wide summary rather than a single-entity or single-signal result.
3. **EDA / overview intent**: Triggered by phrases like "overview," "summary," "explore," or "profile this dataset." Returns total rows, fraud count and rate, a TRANSFER/CASH_OUT-vs-other transaction-type breakdown, and a condensed Signal Validation summary (which heuristics are validated vs. dataset-limited) — computed from the already-loaded working subset and the existing `SIGNAL_VALIDATION` constants, with no re-computation of percentile thresholds.
4. **Behavioral transaction scoring**: Transaction-risk scoring is driven by composite signals rather than a single raw-amount threshold. A dedicated feature-engineering step (`engineer_transaction_features()`) computes sender velocity, rapid cash-out patterns, relative deviation from a sender's typical amount, and structuring-band behavior; `detect_transaction_risk()` then combines those with a large-amount check and weights each signal by its empirically validated predictive value (see Signal Validation).
5. **Explanation**: Each result includes a short rule-based reason that names the signals that drove the score.
6. **Escalation**: The final risk band maps to a simple action recommendation: monitor, review, or report.

## Signal Validation Summary

| Heuristic | Implemented | Validated on PaySim |
|-----------|-------------|---------------------|
| Large Amount | Yes | Yes |
| Velocity | Yes | Dataset limitation |
| Rapid Cash-out | Yes | Dataset limitation |
| Relative Deviation | Yes | Dataset limitation |
| Structuring | Yes | Dataset limitation |

Every heuristic above is implemented; only `large_amount` demonstrated measurable predictive value on PaySim. The other four correspond to legitimate AML behaviors that this synthetic dataset does not represent, not heuristics that fail to work in general. See "Signal Validation" below for the full evaluation methodology, figures, and per-heuristic reasoning.

## Architecture

```
Query → Intent Parser → Router → [EDA Overview | Structuring Detection | Customer Lookup | Feature Engineering → Transaction Risk Scoring | General] → Risk Classification → Explanation Text → Escalation Recommendation → Structured Output
```

### Mapping to the Hackathon Brief

| Brief Component | Implementation |
|---|---|
| EDA Tool | `run_eda()` |
| Feature Engineering Tool | `engineer_transaction_features()` |
| Anomaly Detection Tool | `large_amount` threshold check + composite scoring in `detect_transaction_risk()` |
| Risk Classification Tool | Per-intent `risk_band` mapping logic (low/medium/high thresholds) |
| Explanation Component | Per-intent `reason` text generation (e.g. `detect_transaction_risk()`'s composite-signal explanation) |

## Tech Stack

- **Language**: Python
- **Data processing**: pandas
- **Detection/scoring**: Rule-based heuristics over the PaySim schema
- **Explanation layer**: Programmatic text generation from the selected signals; no external LLM is wired into the current repository
- **Interface**: Streamlit UI for interactive use, with the CLI entry point kept for direct local runs
- **Testing**: pytest

## AI Assistance

- **Claude** (Anthropic) — used for project planning, architecture discussion, dataset inspection tooling, and code review/debugging guidance.
- **GitHub Copilot Chat** — used for in-editor code generation and implementation within VS Code.

## Setup

```bash
git clone https://github.com/san7iya/aml-agent.git
cd aml-agent
pip install -r requirements.txt
```

If the dataset is not already present locally:

1. Download the PaySim CSV from Kaggle.
2. Save it as `Dataset/paysim dataset.csv`.
3. Run the agent once to generate `paysim_working_subset.csv` if it is missing.

## Usage

```bash
streamlit run app.py
```

Opens a single-page UI with example-query buttons for each intent (structuring, customer risk, transaction risk, EDA/overview, general), an Agent Plan panel showing the detected intent/tools/entity, and a color-coded risk result.

Alternatively, use the CLI directly:

```bash
python aml_agent.py "Find structuring patterns in the last 30 days"
python aml_agent.py "Is customer ID C1231006815 suspicious?"
python aml_agent.py "Show me high-risk transactions from the past week"
python aml_agent.py "Profile this dataset"
pytest
```

## Project Structure

```
├── aml_agent.py
├── app.py
├── Dataset/
│   └── paysim dataset.csv  # raw PaySim CSV, kept out of Git
├── tests/
│   └── test_agent.py
├── paysim_working_subset.csv  # generated locally from the raw dataset
├── README.md
├── .gitignore
└── requirements.txt
```

## Known Limitations

The scoring logic is heuristic and tuned for the PaySim schema rather than a production AML program. The dataset is synthetic, so it is useful for demonstration and testing but cannot prove real-world laundering behavior. A large share of labeled fraud in this dataset occurs at a small number of fixed high-value amounts, which is a property of the synthetic data generation rather than a pattern independently discovered by the detection logic. In particular, PaySim commonly injects fraud labels around a small set of round amounts (for example, $10,000,000 appears repeatedly in the dataset), so unusually large round-number transactions should be treated as a synthetic artifact of the generator rather than as a naturally discovered AML pattern.

Independent evaluation against the `isFraud` label (see Signal Validation below) shows this is not simply a matter of further threshold tuning: `large_amount` demonstrated measurable predictive value on PaySim (16.04% recall, 4.75% precision), while `velocity`, `rapid_cash_out`, `relative_deviation`, and `structuring` are correctly implemented but unsupported by this synthetic dataset's characteristics — PaySim does not simulate repeat-sender activity, receive-then-relay mule chains, or structuring/smurfing behavior at all, so no amount of threshold adjustment on this dataset will make those heuristics detect patterns that were never generated in the first place. This is a limitation of the dataset, not evidence that those heuristics are ineffective for real-world AML monitoring. The Streamlit UI is intended for interactive review, while the CLI remains the simplest way to exercise the agent directly from the terminal.

## Signal Validation

`eval_signals.py` independently measures each transaction-risk heuristic against the PaySim ground-truth label (`isFraud`), one signal at a time (not as a composite), by streaming the full raw CSV in chunks (never loading it entirely into memory), filtering to `TRANSFER`/`CASH_OUT` (2,770,409 of 6,362,620 rows), and computing each heuristic's threshold from that population's own distribution (99th percentile of amount, 95th percentile of sender transaction count, 95th percentile of the relative-deviation ratio). Precision, recall, and false-positive rate below are computed over that TRANSFER/CASH_OUT subset; a cross-check confirms all 8,213 labeled frauds in the full dataset fall inside it, so recall figures are not understated by transactions outside the analyzed population. Reproduce with `python eval_signals.py`.

| Signal | Flagged | Flagged % | Fraud Captured | Precision | Recall | FP Count | FP Rate |
|---|---|---|---|---|---|---|---|
| `large_amount` | 27,705 | 1.00% | 1,317 | 4.75% | **16.04%** | 26,388 | 0.96% |
| `velocity` | 3,555 | 0.13% | 16 | 0.45% | 0.19% | 3,539 | 0.13% |
| `rapid_cash_out` | 47 | 0.00% | 0 | n/a | 0.00% | 47 | 0.00% |
| `relative_deviation` | 178 | 0.01% | 0 | n/a | 0.00% | 178 | 0.01% |
| `structuring_strict` (≥3 txns in $9k–$10k) | 0 | 0.00% | 0 | n/a | 0.00% | 0 | 0.00% |
| `structuring_relaxed` (≥2 txns in $8k–$10k) | 0 | 0.00% | 0 | n/a | 0.00% | 0 | 0.00% |

**Implemented vs. validated on PaySim** — every heuristic below is implemented correctly; the table distinguishes that from whether PaySim's data actually contains the pattern each one is designed to catch:

| Heuristic | Implemented | Validated on PaySim | Why |
|---|---|---|---|
| `large_amount` | Yes | **Yes** | Only heuristic with measurable predictive value — real, reproducible correlation with `isFraud`, even though PaySim's amount skew is itself partly a generator artifact. |
| `velocity` | Yes | No | ~95% of PaySim senders transact exactly once; the repeat-offender pattern this targets isn't present in the data. |
| `rapid_cash_out` | Yes | No | PaySim doesn't chain a fraudulent `TRANSFER`'s destination into a later `CASH_OUT`'s source at the ledger level, so the receive-then-relay mule pattern essentially never occurs. |
| `relative_deviation` | Yes | No | Requires ≥2 transactions per sender to establish a baseline; almost no senders qualify. |
| `structuring` (strict & relaxed) | Yes | No | Zero matches dataset-wide — PaySim's fraud generator does not simulate structuring/smurfing behavior at all. |

`large_amount` is weighted as the dominant contributor to `transaction_risk`'s composite score specifically because it was the only heuristic that demonstrated measurable predictive value in this empirical evaluation (see `SIGNAL_VALIDATION` in `aml_agent.py`) — the weighting reflects evidence gathered from `eval_signals.py`, not an assumption that amount-based detection is inherently superior. The other four heuristics are kept in the pipeline at their original weight, not because they proved effective on PaySim, but to preserve a modular, extensible detection architecture: `velocity`, `rapid_cash_out`, `relative_deviation`, and `structuring` target real AML behaviors that this synthetic dataset doesn't represent, and would regain influence automatically — without any code changes beyond re-running `eval_signals.py` against the new data — on a dataset, or in production, where those patterns actually occur.

**Scope note on raw counts**: the table above is computed by `eval_signals.py` directly against the full 6,362,620-row raw CSV (2,770,409 TRANSFER/CASH_OUT rows). The CLI's `transaction_risk` path instead runs against `paysim_working_subset.csv` — a much smaller, fraud-enriched sample (all labeled frauds plus a capped non-fraud sample; see `build_working_subset()`). Both use the identical `large_amount` threshold (`SIGNAL_VALIDATION['large_amount']['threshold']`), so a `flagged_count` you see from the CLI will legitimately differ from the 27,705 figure above — same signal, same threshold, different (and much smaller, much more fraud-dense) population — not an inconsistency in the logic.

**Scope note**: this evaluation covers only these six behavioral heuristics (amount, timing, and sender-history patterns). It does not test balance-based artifacts (e.g. `oldbalanceOrg`/`newbalanceOrig` mismatches) or other transaction-specific characteristics already present in the schema, and should not be read as evidence that such features lack predictive value on PaySim — only that these six specific heuristics do or don't.

## What Makes This Solution Stand Out

The code does not run a fixed pipeline for every prompt. It routes structuring, customer, transaction, EDA/overview, and general questions through different logic paths, which keeps behavior tied to the query intent. The project is also easy to validate because the tests cover routing, dispatch, and subset building against a small sample dataset.
