# AI-Powered Suspicious Activity Detection Agent

An AML (Anti-Money Laundering) suspicious activity detection agent for retail banking. It accepts natural-language queries, routes them to the relevant analysis logic based on intent, and returns risk-scored findings with a short explanation and escalation recommendation.

## Problem Statement

Financial institutions run rule-based AML compliance systems that generate excessive false positives, overwhelming compliance teams, while techniques like structuring and smurfing evade conventional detection. This project builds a query-driven agent that performs targeted analysis — not a fixed pipeline — parsing user intent to invoke only the tools relevant to that specific query (e.g. a structuring query triggers pattern detection; a single-customer query skips full EDA and does a direct lookup).

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

1. **Intent parsing**: The query is normalized and classified into one of four intents: structuring, customer risk, transaction risk, or general.
2. **Dynamic routing**: The agent chooses a different analysis path depending on that intent. Structuring queries run band-based pattern detection, customer queries run direct entity lookups, and transaction-risk queries focus on transfer/cash-out behavior.
3. **Risk scoring**: Scores are computed from concrete signals such as transaction frequency, amount band repetition, balance mismatches, and high-value activity.
4. **Explanation**: Each result includes a short rule-based reason that names the signals that drove the score.
5. **Escalation**: The final risk band maps to a simple action recommendation: monitor, review, or report.

## Architecture

```
Query → Intent Parser → Router → [Structuring Detection | Customer Lookup | Transaction Risk Scoring] → Risk Classification → Explanation Text → Escalation Recommendation → Structured Output
```

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
python aml_agent.py "Find structuring patterns in the last 30 days"
python aml_agent.py "Is customer ID C1231006815 suspicious?"
pytest
```

The CLI entry point is available in `aml_agent.py`. When the Streamlit UI is present in the workspace, launch it with `streamlit run app.py`.

## Project Structure

```
├── aml_agent.py
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

The scoring logic is heuristic and tuned for the PaySim schema rather than a production AML program. The dataset is synthetic, so it is useful for demonstration and testing but cannot prove real-world laundering behavior. The Streamlit UI is intended for interactive review, while the CLI remains the simplest way to exercise the agent directly from the terminal.

## What Makes This Solution Stand Out

The code does not run a fixed pipeline for every prompt. It routes structuring, customer, and transaction questions through different logic paths, which keeps behavior tied to the query intent. The project is also easy to validate because the tests cover routing, dispatch, and subset building against a small sample dataset.
