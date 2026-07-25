"""Streamlit UI for the AML suspicious-activity detection agent.

Thin presentation layer only: every result comes from aml_agent.analyze_query().
No routing, scoring, or detection logic is reimplemented here.
"""

import streamlit as st

from aml_agent import analyze_query

EXAMPLE_QUERIES = [
    ("Structuring", "Find structuring patterns in the last 30 days"),
    ("Customer risk", "Is customer ID C1231006815 suspicious?"),
    ("Transaction risk", "Show me high-risk transactions from the past week"),
    ("EDA / Overview", "Profile this dataset"),
    ("General", "What's going on lately?"),
]

RISK_BAND_DISPLAY = {
    "low": st.success,
    "medium": st.warning,
    "high": st.error,
}

ESCALATION_LABEL = {
    "monitor": "MONITOR",
    "review": "REVIEW",
    "report": "REPORT",
}

st.set_page_config(page_title="AML Suspicious Activity Detection Agent", layout="centered")

if "query" not in st.session_state:
    st.session_state.query = EXAMPLE_QUERIES[0][1]
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False


def use_example(text: str) -> None:
    st.session_state.query = text
    st.session_state.auto_run = True


st.title("AML Suspicious Activity Detection Agent")
st.caption(
    "Natural-language queries are routed by intent (structuring, customer risk, "
    "transaction risk, EDA/overview, or general) to the matching analysis path in aml_agent.py."
)

st.write("Try an example query:")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, (label, text) in zip(example_cols, EXAMPLE_QUERIES):
    col.button(label, on_click=use_example, args=(text,), width="stretch")

with st.form("query_form"):
    query = st.text_input("Query", key="query")
    submitted = st.form_submit_button("Run")

should_run = submitted or st.session_state.auto_run
st.session_state.auto_run = False

if should_run:
    if not query or not query.strip():
        st.warning("Enter a query first.")
    else:
        try:
            with st.spinner("Running analyze_query..."):
                output = analyze_query(query)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error("The agent couldn't process that query. Try a different phrasing.")
            with st.expander("Error details"):
                st.exception(exc)
        else:
            plan = output["plan"]
            result = output["result"]
            escalation = output["escalation"]

            with st.container(border=True):
                st.subheader("Agent Plan")
                # Plain text instead of st.metric: st.metric truncates long values with an
                # ellipsis at narrow column widths (observed on "customer_risk" and
                # "transaction_risk"), and there's no built-in way to disable that -- st.write
                # wraps instead of truncating, regardless of intent-name length.
                st.write(f"**Intent:** {plan['intent']}")
                st.write(f"**Entity ID:** {plan['entity_id'] or '-'}")
                st.write(f"**Rows analyzed:** {output['dataset_rows']:,}")
                st.write("**Tools invoked:** " + ", ".join(plan["tools"]))
                st.caption(f'Query: "{plan["query"]}"')

            st.subheader("Result")
            risk_band = result.get("risk_band", "low")
            band_display = RISK_BAND_DISPLAY.get(risk_band, st.info)
            band_display(f'Risk score: {result.get("risk_score")}  |  Risk band: {risk_band.upper()}')
            # st.text (not st.write/markdown) so literal characters like "$9,000-$10,000" in
            # reason strings can never be misparsed as inline LaTeX math by the markdown renderer.
            st.text(result.get("reason", ""))

            escalation_display = RISK_BAND_DISPLAY.get(risk_band, st.info)
            escalation_display(f'Escalation: {ESCALATION_LABEL.get(escalation, escalation)}')

            flagged_transactions = result.get("flagged_transactions")
            if flagged_transactions:
                st.subheader(f'Flagged Transactions ({result.get("flagged_count", len(flagged_transactions))})')
                # Flatten each row's nested "signals" dict into readable columns instead of
                # letting st.dataframe render it as a raw Python dict string.
                display_rows = [
                    {
                        "step": item["step"],
                        "type": item["type"],
                        "amount": item["amount"],
                        "nameOrig": item["nameOrig"],
                        "nameDest": item["nameDest"],
                        "isFraud": item["isFraud"],
                        "composite_score": item.get("signals", {}).get("composite_score"),
                        "contributing_signals": ", ".join(item.get("signals", {}).get("contributing_signals", [])) or "-",
                    }
                    for item in flagged_transactions
                ]
                st.dataframe(display_rows, width="stretch")

            matched_senders = result.get("matched_senders")
            if matched_senders:
                st.subheader(f"Matched Senders ({len(matched_senders)})")
                st.dataframe(matched_senders, width="stretch")

            if "total_rows" in result and "fraud_rate" in result:
                st.subheader("Dataset Overview")
                overview_cols = st.columns(3)
                overview_cols[0].metric("Total Rows", f'{result["total_rows"]:,}')
                overview_cols[1].metric("Fraud Count", f'{result.get("fraud_count", 0):,}')
                # Display-only rounding to 1 decimal place; result["fraud_rate"] itself is untouched.
                overview_cols[2].metric("Fraud Rate", f'{result["fraud_rate"]:.1%}')

            type_breakdown = result.get("type_breakdown")
            if type_breakdown:
                st.subheader("Transaction Type Breakdown")
                breakdown_cols = st.columns(len(type_breakdown))
                for col, (label, count) in zip(breakdown_cols, type_breakdown.items()):
                    col.metric(label, f"{count:,}")

            signal_validation_summary = result.get("signal_validation_summary")
            if signal_validation_summary:
                st.subheader("Signal Validation Summary")
                st.dataframe(signal_validation_summary, width="stretch")