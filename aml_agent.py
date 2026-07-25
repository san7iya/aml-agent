from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'Dataset' / 'paysim dataset.csv'
WORKING_SUBSET_PATH = ROOT / 'paysim_working_subset.csv'


def build_working_subset(output_path: Path | str | None = None, sample_size: int = 50000, data_path: Path | str | None = None) -> Path:
    output_path = Path(output_path or WORKING_SUBSET_PATH)
    data_path = Path(data_path or DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    fraud_frames = []
    non_fraud_frames = []
    collected_non_fraud = 0

    for chunk in pd.read_csv(data_path, chunksize=250000):
        fraud_chunk = chunk[chunk['isFraud'] == 1]
        if not fraud_chunk.empty:
            fraud_frames.append(fraud_chunk)

        if collected_non_fraud < sample_size:
            non_fraud_chunk = chunk[chunk['isFraud'] == 0]
            if non_fraud_chunk.empty:
                continue

            remaining = sample_size - collected_non_fraud
            if len(non_fraud_chunk) <= remaining:
                non_fraud_frames.append(non_fraud_chunk)
                collected_non_fraud += len(non_fraud_chunk)
            else:
                sampled = non_fraud_chunk.sample(n=remaining, random_state=42)
                non_fraud_frames.append(sampled)
                collected_non_fraud += len(sampled)

    if not fraud_frames and not non_fraud_frames:
        raise ValueError(f'No rows were loaded from {data_path}')

    subset = pd.concat([*fraud_frames, *non_fraud_frames], axis=0, ignore_index=True)
    subset.to_csv(output_path, index=False)
    return output_path


def load_working_subset(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or WORKING_SUBSET_PATH)
    if not path.exists():
        path = build_working_subset(path)
    return pd.read_csv(path)


def build_plan(query: str) -> Dict[str, Any]:
    text = query.lower()

    if 'structur' in text or 'smurf' in text or 'pattern' in text:
        intent = 'structuring'
        tools = ['feature_engineering', 'anomaly_detection', 'risk_scoring']
    elif 'customer' in text or 'customer id' in text or 'account' in text or 'sender' in text:
        intent = 'customer_risk'
        tools = ['customer_profile', 'risk_scoring']
    elif 'transaction' in text or 'high-risk' in text or 'flag' in text or 'suspicious' in text or 'risk' in text:
        intent = 'transaction_risk'
        tools = ['risk_scoring']
    else:
        intent = 'general'
        tools = ['risk_scoring']

    entity_match = re.search(r'\b(?:customer|account|sender|id)\s*(?:id\s*)?([A-Za-z]?\d+)\b', query, re.I)
    entity_id = entity_match.group(1) if entity_match else None

    return {
        'intent': intent,
        'tools': tools,
        'entity_id': entity_id,
        'query': query,
    }


def detect_structuring(df: pd.DataFrame) -> Dict[str, Any]:
    band_mask = df['amount'].between(9000, 10000, inclusive='both')
    structuring_rows = df[band_mask].copy()
    sender_counts = structuring_rows.groupby('nameOrig').size().reset_index(name='band_count')
    sender_counts = sender_counts[sender_counts['band_count'] >= 3].sort_values('band_count', ascending=False)

    if sender_counts.empty:
        return {
            'matched_senders': [],
            'risk_score': 0.2,
            'risk_band': 'low',
            'reason': 'No sender showed 3+ transactions in the $9,000-$10,000 band.',
        }

    matched_senders = []
    for _, row in sender_counts.iterrows():
        sender_rows = df[df['nameOrig'] == row['nameOrig']]
        matched_senders.append({
            'sender': row['nameOrig'],
            'band_count': int(row['band_count']),
            'total_transactions': int(len(sender_rows)),
        })

    band_count = int(sender_counts['band_count'].sum())
    risk_score = min(1.0, 0.35 + 0.12 * len(matched_senders) + 0.08 * band_count)
    if risk_score >= 0.8:
        band = 'high'
    elif risk_score >= 0.55:
        band = 'medium'
    else:
        band = 'low'

    return {
        'matched_senders': matched_senders,
        'risk_score': round(risk_score, 3),
        'risk_band': band,
        'reason': f'Detected {len(matched_senders)} sender(s) with repeated $9,000-$10,000 band activity, indicating possible structuring.',
    }


def detect_transaction_risk(df: pd.DataFrame) -> Dict[str, Any]:
    high_value_mask = (df['type'].isin(['TRANSFER', 'CASH_OUT'])) & (df['amount'] >= 50000)
    mismatch_mask = (
        (df['oldbalanceOrg'] - df['newbalanceOrig']).abs() > 1
    ) | (
        (df['oldbalanceDest'] - df['newbalanceDest']).abs() > 1
    )

    flagged_rows = df[high_value_mask | mismatch_mask].copy()
    flagged_rows = flagged_rows.sort_values('amount', ascending=False)

    if flagged_rows.empty:
        return {
            'flagged_transactions': [],
            'risk_score': 0.2,
            'risk_band': 'low',
            'reason': 'No high-value transfer or balance-mismatch transactions were detected.',
        }

    fraud_count = int(flagged_rows['isFraud'].sum())
    big_txn_count = int((flagged_rows['amount'] >= 50000).sum())
    mismatch_count = int(mismatch_mask.sum())
    risk_score = min(1.0, 0.2 + 0.15 * big_txn_count + 0.2 * mismatch_count + 0.1 * fraud_count)
    if risk_score >= 0.8:
        band = 'high'
    elif risk_score >= 0.55:
        band = 'medium'
    else:
        band = 'low'

    return {
        'flagged_transactions': flagged_rows[['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud']].head(5).to_dict(orient='records'),
        'risk_score': round(risk_score, 3),
        'risk_band': band,
        'reason': f'Detected {big_txn_count} high-value transaction(s) and {mismatch_count} balance-mismatch signal(s).',
    }


def score_customer(df: pd.DataFrame, entity_id: str) -> Dict[str, Any]:
    entity_id = str(entity_id)
    if entity_id.isdigit():
        customer_mask = df['nameOrig'].astype(str).str.endswith(entity_id)
    else:
        customer_mask = df['nameOrig'].astype(str) == entity_id

    customer_rows = df[customer_mask]
    if customer_rows.empty:
        return {'entity_id': entity_id, 'risk_score': 0.0, 'risk_band': 'low', 'reason': 'No matching customer activity found.'}

    fraud_count = int(customer_rows['isFraud'].sum())
    total_count = len(customer_rows)
    velocity = int((customer_rows['amount'] > 10000).sum())
    balance_mismatch = int(((customer_rows['oldbalanceOrg'] - customer_rows['newbalanceOrig']).abs() > 1).sum())
    risk_score = min(1.0, 0.2 + 0.25 * fraud_count + 0.05 * velocity + 0.15 * balance_mismatch)
    if risk_score >= 0.8:
        band = 'high'
    elif risk_score >= 0.55:
        band = 'medium'
    else:
        band = 'low'

    return {
        'entity_id': entity_id,
        'risk_score': round(risk_score, 3),
        'risk_band': band,
        'reason': f'{fraud_count} fraud transaction(s), {velocity} high-value activity event(s), and {balance_mismatch} balance-mismatch signal(s).',
    }


def analyze_query(query: str, data_path: Path | str | None = None) -> Dict[str, Any]:
    plan = build_plan(query)
    df = load_working_subset(data_path)

    if plan['intent'] == 'structuring':
        result = detect_structuring(df)
    elif plan['intent'] == 'transaction_risk':
        result = detect_transaction_risk(df)
    elif plan['intent'] == 'customer_risk' and plan['entity_id'] is not None:
        result = score_customer(df, plan['entity_id'])
    else:
        result = {
            'entity_id': None,
            'risk_score': 0.25,
            'risk_band': 'low',
            'reason': 'No specific pattern was detected from the query terms.',
        }

    return {
        'plan': plan,
        'result': result,
        'dataset_rows': len(df),
        'escalation': 'monitor' if result['risk_band'] == 'low' else 'review' if result['risk_band'] == 'medium' else 'report',
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a natural-language AML query against the PaySim dataset.')
    parser.add_argument('query', nargs='?', default='Find structuring patterns in the last 30 days')
    args = parser.parse_args()

    output = analyze_query(args.query)
    print(output)
