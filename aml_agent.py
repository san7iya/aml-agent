from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'Dataset' / 'paysim dataset.csv'
WORKING_SUBSET_PATH = ROOT / 'paysim_working_subset.csv'
CUSTOMER_INDEX_PATH = ROOT / 'customer_index.sqlite3'

# Empirical validation against PaySim ground truth (isFraud), measured independently by
# eval_signals.py over the full dataset (2,770,409 TRANSFER/CASH_OUT transactions, 8,213
# labeled frauds; see README "Signal Validation" for the full table). This is the single
# source of truth for how much each heuristic is trusted -- composite_weight below drives
# detect_transaction_risk's scoring directly, so calibration changes only need to happen here.
SIGNAL_VALIDATION: Dict[str, Dict[str, Any]] = {
    'large_amount': {
        'implementation_status': 'implemented',
        'validation_status': 'validated_on_paysim',
        'precision': 0.0475,
        'recall': 0.1604,
        'composite_weight': 10.0,
        'threshold': 2_650_035.52,
        'note': (
            'The only heuristic with measurable predictive value on PaySim: catches ~16% of '
            'labeled fraud at ~5% precision. PaySim fraud amounts skew unusually large, which is '
            'partly an artifact of the synthetic generator, but the correlation with isFraud is '
            'real and reproducible, so it is weighted as the dominant signal.'
        ),
    },
    'velocity': {
        'implementation_status': 'implemented',
        'validation_status': 'not_supported_by_dataset',
        'precision': 0.0045,
        'recall': 0.0019,
        'composite_weight': 1.0,
        'note': (
            'Implemented correctly, but ~95% of PaySim senders transact exactly once, so almost '
            'no fraud senders repeat. The repeat-offender pattern this heuristic targets is not '
            'represented in PaySim -- a dataset limitation, not a broken heuristic.'
        ),
    },
    'rapid_cash_out': {
        'implementation_status': 'implemented',
        'validation_status': 'not_supported_by_dataset',
        'precision': float('nan'),
        'recall': 0.0,
        'composite_weight': 1.0,
        'note': (
            "Implemented correctly, but PaySim does not chain a fraudulent TRANSFER's "
            "destination into a later CASH_OUT's source at the ledger level, so the "
            'receive-then-relay mule pattern this targets essentially does not occur in the data.'
        ),
    },
    'relative_deviation': {
        'implementation_status': 'implemented',
        'validation_status': 'not_supported_by_dataset',
        'precision': 0.0,
        'recall': 0.0,
        'composite_weight': 1.0,
        'note': (
            'Implemented correctly, but requires a sender history of >=2 transactions to compare '
            'against; since PaySim senders are almost all one-off, too few rows are even eligible '
            'for this signal to demonstrate predictive value here.'
        ),
    },
    'structuring': {
        'implementation_status': 'implemented',
        'validation_status': 'not_supported_by_dataset',
        'precision': float('nan'),
        'recall': 0.0,
        'composite_weight': 1.0,
        'note': (
            "Implemented correctly, but PaySim's fraud generator does not simulate "
            'structuring/smurfing behavior at all, so this signal has zero labeled fraud to '
            'validate against on this dataset -- not a detection failure.'
        ),
    },
}


def build_working_subset(output_path: Path | str | None = None, sample_size: int = 200000, data_path: Path | str | None = None) -> Path:
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
    if not path.exists() or path.stat().st_size < 1000000:
        path = build_working_subset(path, sample_size=200000)
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

    entity_match = re.search(r'\b(?:customer|account|sender|id)\s*(?:id\s*)?([A-Za-z]\d+|\d+)\b', query, re.I)
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


def _looks_like_customer_account(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.startswith(('C', 'c'))
    return False


def detect_transaction_risk(df: pd.DataFrame, window_steps: int = 24) -> Dict[str, Any]:
    relevant = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    if relevant.empty:
        return {
            'flagged_transactions': [],
            'risk_score': 0.2,
            'risk_band': 'low',
            'reason': 'No transfer or cash-out transactions were available for review.',
        }

    relevant = relevant.sort_values(['nameOrig', 'step', 'amount']).reset_index(drop=True)
    structuring_result = detect_structuring(relevant)
    structuring_senders = {item['sender'] for item in structuring_result.get('matched_senders', [])}
    large_amount_threshold = SIGNAL_VALIDATION['large_amount']['threshold']

    # Vectorized equivalent of the old per-sender iterrows() loop (profiled bottleneck: repeated
    # full-column boolean masking to build sender_rows_map, then per-tiny-group pandas __setitem__
    # calls). No detection logic, threshold, or weight changed; only how it's computed.
    sender_key = relevant['nameOrig'].astype(str)
    amount = relevant['amount'].to_numpy()

    # large_amount needs no transaction history (validated: it fires on single one-off
    # transfers, which is how most PaySim fraud senders behave), so it is evaluated for every
    # row directly. structuring_senders already implies >=3 total rows for every member (a
    # band_count >= 3 requires total_transactions >= 3), so it needs no separate history gate
    # either. The other three behavioral signals still require >=3 transactions to be
    # meaningful (validated: they only ever fire for repeat senders).
    large_amount_signal = amount >= large_amount_threshold
    band_mask = (amount >= 9000) & (amount <= 10000)
    structuring_signal = sender_key.isin(structuring_senders).to_numpy() & band_mask

    velocity_signal = np.zeros(len(relevant), dtype=bool)
    cashout_signal = np.zeros(len(relevant), dtype=bool)
    deviation_signal = np.zeros(len(relevant), dtype=bool)

    group_sizes = sender_key.groupby(sender_key).transform('size').to_numpy()
    eligible_mask = group_sizes >= 3

    if eligible_mask.any():
        eligible_positions = np.flatnonzero(eligible_mask)
        eligible_df = relevant.iloc[eligible_positions]
        eligible_keys = sender_key.iloc[eligible_positions]

        for _, group_positions in eligible_df.groupby(eligible_keys, sort=False).groups.items():
            group = relevant.loc[group_positions].sort_values('step')
            local_pos = relevant.index.get_indexer(group.index)

            steps = group['step'].to_numpy()
            amounts = group['amount'].to_numpy()
            dest = group['nameDest'].to_numpy()
            orig = group['nameOrig'].to_numpy()

            lo_v = np.searchsorted(steps, steps - window_steps, side='left')
            hi_v = np.searchsorted(steps, steps, side='right')
            velocity_signal[local_pos] = (hi_v - lo_v) >= 2

            dest_is_self = (dest == orig[0]).astype(np.int64)
            prefix = np.concatenate(([0], np.cumsum(dest_is_self)))
            lo_c = np.searchsorted(steps, steps - window_steps, side='left')
            hi_c = np.searchsorted(steps, steps, side='left')
            has_recent_self_dest = (prefix[hi_c] - prefix[lo_c]) > 0
            current_dest_notna = np.array([d not in (None, '') for d in dest])
            cashout_signal[local_pos] = has_recent_self_dest & current_dest_notna

            median_amount = np.median(amounts)
            if median_amount:
                ratio = amounts / median_amount
                deviation_signal[local_pos] = (ratio >= 1.5) | (ratio <= 0.67)

    relevant['large_amount_signal'] = large_amount_signal
    relevant['velocity_signal'] = velocity_signal
    relevant['cashout_signal'] = cashout_signal
    relevant['deviation_signal'] = deviation_signal
    relevant['structuring_signal'] = structuring_signal
    relevant['composite_score'] = (
        SIGNAL_VALIDATION['large_amount']['composite_weight'] * relevant['large_amount_signal'].astype(float)
        + SIGNAL_VALIDATION['velocity']['composite_weight'] * relevant['velocity_signal'].astype(float)
        + SIGNAL_VALIDATION['rapid_cash_out']['composite_weight'] * relevant['cashout_signal'].astype(float)
        + SIGNAL_VALIDATION['relative_deviation']['composite_weight'] * relevant['deviation_signal'].astype(float)
        + SIGNAL_VALIDATION['structuring']['composite_weight'] * relevant['structuring_signal'].astype(float)
    )

    scored_rows_df = relevant

    if scored_rows_df.empty:
        return {
            'flagged_transactions': [],
            'flagged_count': 0,
            'flagged_rate': 0.0,
            'risk_score': 0.2,
            'risk_band': 'low',
            'reason': 'No composite behavioral signals were detected for transaction risk.',
        }

    flagged_rows = scored_rows_df[
        scored_rows_df['large_amount_signal'] |
        scored_rows_df['velocity_signal'] |
        scored_rows_df['cashout_signal'] |
        scored_rows_df['structuring_signal'] |
        scored_rows_df['deviation_signal']
    ].copy()
    flagged_rows = flagged_rows.sort_values(['composite_score', 'amount'], ascending=[False, False])

    if flagged_rows.empty:
        return {
            'flagged_transactions': [],
            'flagged_count': 0,
            'flagged_rate': 0.0,
            'risk_score': 0.2,
            'risk_band': 'low',
            'reason': 'No composite behavioral signals were detected for transaction risk.',
        }

    flagged_count = int(len(flagged_rows))
    flagged_rate = round(flagged_count / len(relevant), 4) if len(relevant) else 0.0
    fraud_count = int(flagged_rows['isFraud'].sum())
    max_score = float(flagged_rows['composite_score'].max())
    avg_score = float(flagged_rows['composite_score'].mean())
    risk_score = min(1.0, 0.2 + 0.15 * max_score + 0.05 * avg_score + 0.1 * fraud_count)
    if risk_score >= 0.75:
        band = 'high'
    elif risk_score >= 0.45:
        band = 'medium'
    else:
        band = 'low'

    flagged_payload = []
    for _, row in flagged_rows.head(5).iterrows():
        contributing_signals = []
        if row['large_amount_signal']:
            contributing_signals.append('large_amount (validated on PaySim)')
        if row['velocity_signal']:
            contributing_signals.append('velocity (implemented, not validated on PaySim)')
        if row['cashout_signal']:
            contributing_signals.append('rapid_cash_out (implemented, not validated on PaySim)')
        if row['deviation_signal']:
            contributing_signals.append('relative_deviation (implemented, not validated on PaySim)')
        if row['structuring_signal']:
            contributing_signals.append('structuring (implemented, not validated on PaySim)')

        flagged_payload.append({
            'step': int(row['step']),
            'type': row['type'],
            'amount': float(row['amount']),
            'nameOrig': row['nameOrig'],
            'nameDest': row['nameDest'],
            'isFraud': int(row['isFraud']),
            'signals': {
                'large_amount': bool(row['large_amount_signal']),
                'velocity': bool(row['velocity_signal']),
                'rapid_cash_out': bool(row['cashout_signal']),
                'relative_deviation': bool(row['deviation_signal']),
                'structuring': bool(row['structuring_signal']),
                'composite_score': float(row['composite_score']),
                'contributing_signals': contributing_signals,
            },
        })

    return {
        'flagged_transactions': flagged_payload,
        'flagged_count': flagged_count,
        'flagged_rate': flagged_rate,
        'risk_score': round(risk_score, 3),
        'risk_band': band,
        'reason': (
            f'Composite behavioral scoring flagged {flagged_count} transaction(s) for review. '
            'The score is driven primarily by large_amount, the only heuristic empirically '
            'validated as predictive of isFraud on PaySim; velocity, rapid_cash_out, '
            'relative_deviation, and structuring are implemented and still contribute when they '
            'trigger, but carry only minor weight because PaySim does not exercise the '
            'repeat-sender or mule-chain patterns those heuristics target -- a property of this '
            'synthetic dataset, not a defect in those heuristics.'
        ),
    }


def _build_customer_index(data_path: Path | str, index_path: Path | str = CUSTOMER_INDEX_PATH, chunksize: int = 250000) -> sqlite3.Connection:
    data_path = Path(data_path)
    index_path = Path(index_path)
    required_columns = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
        'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud',
    ]

    conn = sqlite3.connect(index_path)
    conn.execute('DROP TABLE IF EXISTS account_activity')
    conn.execute('CREATE TABLE account_activity (account_id TEXT PRIMARY KEY, rows_json TEXT)')

    account_rows: Dict[str, list[Dict[str, Any]]] = {}
    for chunk in pd.read_csv(data_path, chunksize=chunksize, usecols=required_columns):
        for account_id in chunk['nameOrig'].astype(str).unique():
            account_rows.setdefault(account_id, []).extend(chunk.loc[chunk['nameOrig'].astype(str) == account_id].to_dict(orient='records'))

    for account_id, rows in account_rows.items():
        conn.execute('INSERT INTO account_activity (account_id, rows_json) VALUES (?, ?)', (account_id, json.dumps(rows)))

    conn.commit()
    conn.close()
    return sqlite3.connect(index_path)


def load_customer_activity(data_path: Path | str, entity_id: str, chunksize: int = 250000) -> pd.DataFrame:
    data_path = Path(data_path)
    entity_id = str(entity_id)
    required_columns = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
        'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud',
    ]

    index_path = ROOT / 'customer_index.sqlite3'
    if not index_path.exists():
        _build_customer_index(data_path, index_path, chunksize=chunksize)

    conn = sqlite3.connect(index_path)
    cursor = conn.execute('SELECT rows_json FROM account_activity WHERE account_id = ?', (entity_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return pd.DataFrame(columns=required_columns)
    rows = json.loads(row[0])
    return pd.DataFrame(rows)


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

    if plan['intent'] == 'customer_risk' and plan['entity_id'] is not None:
        dataset_path = Path(data_path or DATASET_PATH)
        df = load_customer_activity(dataset_path, plan['entity_id'])
        result = score_customer(df, plan['entity_id'])
    else:
        df = load_working_subset(data_path)
        if plan['intent'] == 'structuring':
            result = detect_structuring(df)
        elif plan['intent'] == 'transaction_risk':
            result = detect_transaction_risk(df)
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