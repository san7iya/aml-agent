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
    ("General", "Give me a general overview of recent activity"),
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
    "transaction risk, or general) to the matching analysis path in aml_agent.py."
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
                plan_cols = st.columns(3)
                plan_cols[0].metric("Intent", plan["intent"])
                plan_cols[1].metric("Entity ID", plan["entity_id"] or "-")
                plan_cols[2].metric("Rows analyzed", f'{output["dataset_rows"]:,}')
                st.write("**Tools invoked:** " + ", ".join(plan["tools"]))
                st.caption(f'Query: "{plan["query"]}"')

            st.subheader("Result")
            risk_band = result.get("risk_band", "low")
            band_display = RISK_BAND_DISPLAY.get(risk_band, st.info)
            band_display(f'Risk score: {result.get("risk_score")}  |  Risk band: {risk_band.upper()}')
            st.write(result.get("reason", ""))

            escalation_display = RISK_BAND_DISPLAY.get(risk_band, st.info)
            escalation_display(f'Escalation: {ESCALATION_LABEL.get(escalation, escalation)}')

            flagged_transactions = result.get("flagged_transactions")
            if flagged_transactions:
                st.subheader(f'Flagged Transactions ({result.get("flagged_count", len(flagged_transactions))})')
                st.dataframe(flagged_transactions, width="stretch")

            matched_senders = result.get("matched_senders")
            if matched_senders:
                st.subheader(f"Matched Senders ({len(matched_senders)})")
                st.dataframe(matched_senders, width="stretch")