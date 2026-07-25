# AI-Powered Suspicious Activity Detection Agent

An agentic system for AML (Anti-Money Laundering) suspicious activity detection in retail banking. Accepts natural-language queries, dynamically decides which analysis tools to invoke based on query intent, and returns risk-scored, explained findings with an escalation recommendation.

## Problem Statement

Financial institutions run rule-based AML compliance systems that generate excessive false positives, overwhelming compliance teams, while techniques like structuring and smurfing evade conventional detection. This project builds a query-driven agent that performs targeted analysis — not a fixed pipeline — parsing user intent to invoke only the tools relevant to that specific query (e.g. a structuring query triggers pattern detection; a single-customer query skips full EDA and does a direct lookup).

<!-- TODO: expand with 2-3 sentences on why this framing (query-driven, tool-selective) matters over a fixed pipeline -->

## Dataset

- **Source**: [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), Kaggle.
- **Nature**: Fully synthetic transaction data simulating mobile money transfers, generated from aggregated real financial logs by the original researchers (not real customer data — no PII, compliant with the "no proprietary/confidential data" rule).
- **Schema**: `step` (time unit, 1 hour), `type` (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER), `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- **Working subset**: A sampled subset (all fraud rows + a capped sample of non-fraud rows) is used during development for iteration speed. <!-- TODO: confirm final sample size used -->

<!-- TODO: if any structuring cases were synthetically injected because PaySim lacked natural examples, disclose that explicitly here — required by the "data fabrication" code-of-conduct clause. State clearly which rows are injected and why. -->

**Citation**: Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *28th European Modeling and Simulation Symposium (EMSS)*.

## Solution Approach

<!-- TODO: 3-5 sentences, plain language -->
1. **Intent parsing**: Query is parsed to extract intent type (structuring / customer risk / transaction risk / general), target entity, and relevant filters.
2. **Dynamic tool dispatch**: Based on intent, only the relevant analysis path is executed — e.g. structuring queries run pattern-detection logic over transaction frequency/amount bands; customer queries run direct entity lookup and profiling; the system does not run every tool for every query.
3. **Risk scoring**: Computed from concrete signals (e.g. transaction velocity, amount deviation, balance-reconciliation mismatches, structuring-band frequency) rather than a single opaque model output.
4. **Explanation**: A human-readable reason is generated for each flag, tied to the specific signals that drove the score.
5. **Escalation**: Risk band (low/medium/high) maps to a recommended action (monitor/review/report).

## Architecture

<!-- TODO: paste/describe diagram — mirrors the 2-slide deck's architecture slide -->
```
Query → Intent Parser → Router (dynamic tool selection) → [Structuring Detection | Customer Profiling | Transaction Risk Scoring] → Risk Classification → Explanation Layer → Escalation Recommendation → Structured Output
```

## Tech Stack

- **Language**: Python
- **Data processing**: pandas
- **Detection/scoring**: <!-- TODO: confirm final approach — rules only / Isolation Forest / hybrid -->
- **Explanation layer**: <!-- TODO: confirm LLM used (e.g. Gemini API, free tier) -->
- **Interface**: Streamlit
- **Testing**: pytest

## AI Assistance Disclosure

Per hackathon rules requiring disclosure of all AI tooling used:
- **Claude** (Anthropic) — used for project planning, architecture discussion, dataset inspection tooling, and code review/debugging guidance.
- **GitHub Copilot Chat** — used for in-editor code generation and implementation within VS Code.

<!-- TODO: add/remove tools to match what was actually used by submission time -->

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

<!-- TODO: add dataset download step — since raw PaySim CSV is likely too large to commit, document where to download it and where to place it -->

## Usage

```bash
# CLI
python aml_agent.py "Find structuring patterns in the last 30 days"
python aml_agent.py "Is customer ID 4521 suspicious?"

# Streamlit app
streamlit run app.py
```

<!-- TODO: confirm actual entry-point filenames match what's in the repo -->

## Project Structure

```
<!-- TODO: fill in once repo structure is finalized -->
├── aml_agent.py
├── app.py                 # Streamlit interface
├── tests/
│   └── test_agent.py
├── Dataset/
│   └── paysim dataset.csv  # not committed if too large — see Setup
├── README.md
└── requirements.txt
```

## Known Limitations

<!-- TODO: be honest here — e.g. synthetic data may not fully reflect real-world laundering patterns; scoring thresholds are heuristic, not calibrated against a validated ground truth beyond PaySim's labels; explanation layer depends on LLM API availability -->

## What Makes This Solution Stand Out

<!-- TODO: this should mirror your deck's differentiation slide — draft 2-3 concrete points once the router is fully working, e.g. "genuinely query-driven tool selection verified against N distinct query types" rather than a fixed pipeline -->
