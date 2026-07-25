"""Standalone evaluation of AML detection heuristics against PaySim ground truth (isFraud).

Each of the six signals below is evaluated independently (not as a composite) so we can see
which raw heuristics actually correlate with the isFraud label on this dataset. This script
does not import or exercise aml_agent.py; thresholds and definitions here are private to it.

Run: python eval_signals.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'Dataset' / 'paysim dataset.csv'

CHUNK_SIZE = 250_000
REQUIRED_COLUMNS = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud']
RELEVANT_TYPES = {'TRANSFER', 'CASH_OUT'}
READ_DTYPES = {'step': 'int32', 'amount': 'float64', 'isFraud': 'int8'}

LARGE_AMOUNT_PERCENTILE = 0.99
VELOCITY_PERCENTILE = 0.95
RELATIVE_DEVIATION_PERCENTILE = 0.95
RELATIVE_DEVIATION_MIN_SENDER_TXNS = 2

# "Rapid cash-out": an account receives funds as nameDest, then later sends funds as
# nameOrig within `window` steps of that receipt. The gap (send_step - most recent prior
# receipt step for that account) is computed ONCE in compute_rapid_cashout_gap() and reused
# for every window below, so the definition never drifts between the main table and the
# sensitivity sweep.
RAPID_CASHOUT_WINDOWS = [1, 3, 6, 12, 24]
RAPID_CASHOUT_DEFAULT_WINDOW = 3

STRUCTURING_STRICT_BAND = (9000.0, 10000.0)
STRUCTURING_STRICT_MIN_COUNT = 3
STRUCTURING_RELAXED_BAND = (8000.0, 10000.0)
STRUCTURING_RELAXED_MIN_COUNT = 2

SIGNAL_NAMES = [
    'large_amount',
    'rapid_cash_out',
    'velocity',
    'relative_deviation',
    'structuring_strict',
    'structuring_relaxed',
]


def load_transfer_cashout_subset(data_path: Path) -> Tuple[pd.DataFrame, int, int]:
    """Stream the CSV in chunks, keeping only TRANSFER/CASH_OUT rows and required columns.

    The full CSV is never held in memory at once: each chunk is read, tallied, filtered, and
    only the (much smaller) TRANSFER/CASH_OUT slice is kept for concatenation.
    """
    frames: List[pd.DataFrame] = []
    rows_processed = 0
    total_fraud_all_types = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=REQUIRED_COLUMNS,
        dtype=READ_DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        rows_processed += len(chunk)
        total_fraud_all_types += int(chunk['isFraud'].sum())

        relevant = chunk[chunk['type'].isin(RELEVANT_TYPES)]
        if not relevant.empty:
            frames.append(relevant.copy())

    if not frames:
        raise ValueError(f'No TRANSFER/CASH_OUT rows found in {data_path}')

    subset = pd.concat(frames, axis=0, ignore_index=True)
    subset['type'] = subset['type'].astype('category')
    return subset, rows_processed, total_fraud_all_types


def compute_sender_aggregates(subset: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    sender_group = subset.groupby('nameOrig')['amount']
    sender_counts = sender_group.size()
    sender_medians = sender_group.median()
    return sender_counts, sender_medians


def compute_rapid_cashout_gap(subset: pd.DataFrame) -> pd.Series:
    """For each row, the step-gap to the sending account's most recent prior receipt.

    NaN means the sending account never received funds (as nameDest) beforehand, i.e. no
    rapid cash-out pattern is possible for that transaction under any window.

    pd.merge_asof requires each frame sorted by its own `on` column (not by [account, step])
    -- confirmed empirically, sorting by [account, step] instead raises "left keys must be
    sorted". Grouping is handled internally via `by`.
    """
    sends = subset[['nameOrig', 'step']].reset_index().rename(
        columns={'index': 'row_id', 'nameOrig': 'account', 'step': 'send_step'}
    )
    receipts = subset[['nameDest', 'step']].rename(columns={'nameDest': 'account', 'step': 'recv_step'})

    sends_sorted = sends.sort_values('send_step').reset_index(drop=True)
    receipts_sorted = receipts.sort_values('recv_step').reset_index(drop=True)

    merged = pd.merge_asof(
        sends_sorted,
        receipts_sorted,
        left_on='send_step',
        right_on='recv_step',
        by='account',
        direction='backward',
    )
    merged['gap'] = merged['send_step'] - merged['recv_step']
    return merged.set_index('row_id')['gap'].reindex(range(len(subset)))


def compute_flags(
    subset: pd.DataFrame,
    sender_counts: pd.Series,
    sender_medians: pd.Series,
    rapid_gap: pd.Series,
    rapid_window: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    amount = subset['amount'].to_numpy()
    name_orig = subset['nameOrig']

    large_amount_threshold = float(np.quantile(amount, LARGE_AMOUNT_PERCENTILE))
    large_amount_flag = amount >= large_amount_threshold

    gap = rapid_gap.to_numpy()
    rapid_cashout_flag = ~np.isnan(gap) & (gap <= rapid_window)

    velocity_threshold = float(sender_counts.quantile(VELOCITY_PERCENTILE))
    sender_count_per_row = name_orig.map(sender_counts).to_numpy()
    velocity_flag = sender_count_per_row > velocity_threshold

    eligible_senders = sender_counts[sender_counts >= RELATIVE_DEVIATION_MIN_SENDER_TXNS].index
    eligible_mask = name_orig.isin(eligible_senders).to_numpy()
    sender_median_per_row = name_orig.map(sender_medians).to_numpy()
    safe_median = np.where(sender_median_per_row > 0, sender_median_per_row, np.nan)
    ratio = np.where(eligible_mask, amount / safe_median, np.nan)
    deviation_threshold = float(np.nanquantile(ratio, RELATIVE_DEVIATION_PERCENTILE))
    deviation_flag = ~np.isnan(ratio) & (ratio >= deviation_threshold)

    band_strict = (amount >= STRUCTURING_STRICT_BAND[0]) & (amount <= STRUCTURING_STRICT_BAND[1])
    strict_band_counts = subset.loc[band_strict].groupby('nameOrig').size()
    strict_senders = strict_band_counts[strict_band_counts >= STRUCTURING_STRICT_MIN_COUNT].index
    structuring_strict_flag = band_strict & name_orig.isin(strict_senders).to_numpy()

    band_relaxed = (amount >= STRUCTURING_RELAXED_BAND[0]) & (amount <= STRUCTURING_RELAXED_BAND[1])
    relaxed_band_counts = subset.loc[band_relaxed].groupby('nameOrig').size()
    relaxed_senders = relaxed_band_counts[relaxed_band_counts >= STRUCTURING_RELAXED_MIN_COUNT].index
    structuring_relaxed_flag = band_relaxed & name_orig.isin(relaxed_senders).to_numpy()

    flags = {
        'large_amount': large_amount_flag,
        'rapid_cash_out': rapid_cashout_flag,
        'velocity': velocity_flag,
        'relative_deviation': deviation_flag,
        'structuring_strict': structuring_strict_flag,
        'structuring_relaxed': structuring_relaxed_flag,
    }
    thresholds = {
        'large_amount_threshold': large_amount_threshold,
        'velocity_threshold': velocity_threshold,
        'relative_deviation_threshold': deviation_threshold,
    }
    return flags, thresholds


def compute_metrics(
    flags: Dict[str, np.ndarray],
    is_fraud: np.ndarray,
    total_frauds: int,
    total_legitimate: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Precision/recall/FPR are computed entirely over the analyzed TRANSFER/CASH_OUT subset:
    total_frauds and total_legitimate are subset-scoped counts, not full-dataset counts.
    A signal with flagged_count == 0 is a valid outcome (precision/recall reported as n/a),
    not an error condition.
    """
    n = len(is_fraud)
    rows = []
    fraud_flags: Dict[str, np.ndarray] = {}

    for name in SIGNAL_NAMES:
        flag = flags[name]
        fraud_hit = flag & is_fraud
        flagged_count = int(flag.sum())
        fraud_captured = int(fraud_hit.sum())
        fp = flagged_count - fraud_captured

        rows.append({
            'signal': name,
            'flagged_count': flagged_count,
            'flagged_pct': flagged_count / n if n else float('nan'),
            'fraud_captured': fraud_captured,
            'precision': fraud_captured / flagged_count if flagged_count else float('nan'),
            'recall': fraud_captured / total_frauds if total_frauds else float('nan'),
            'false_positive_count': fp,
            'false_positive_rate': fp / total_legitimate if total_legitimate else float('nan'),
        })
        fraud_flags[name] = fraud_hit

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(['recall', 'precision'], ascending=[False, False]).reset_index(drop=True)
    return results_df, fraud_flags


def compute_overlap(fraud_flags: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    matrix = pd.DataFrame(index=SIGNAL_NAMES, columns=SIGNAL_NAMES, dtype='int64')
    for a in SIGNAL_NAMES:
        for b in SIGNAL_NAMES:
            matrix.loc[a, b] = int((fraud_flags[a] & fraud_flags[b]).sum())

    unique_counts: Dict[str, int] = {}
    for a in SIGNAL_NAMES:
        others = [b for b in SIGNAL_NAMES if b != a]
        caught_by_any_other = np.zeros_like(fraud_flags[a])
        for b in others:
            caught_by_any_other |= fraud_flags[b]
        unique_counts[a] = int((fraud_flags[a] & ~caught_by_any_other).sum())

    return matrix, unique_counts


def evaluate_rapid_cashout_windows(
    rapid_gap: pd.Series,
    is_fraud: np.ndarray,
    windows: List[int],
    total_frauds: int,
    total_legitimate: int,
) -> pd.DataFrame:
    """Reuses the single precomputed gap array for every window -- the rapid cash-out
    definition (receive as nameDest, then send as nameOrig within N steps) never changes
    between window settings, only the cutoff applied to the same gap values.
    """
    gap = rapid_gap.to_numpy()
    has_match = ~np.isnan(gap)
    n = len(gap)
    rows = []
    for window in windows:
        flag = has_match & (gap <= window)
        flagged_count = int(flag.sum())
        fraud_captured = int((flag & is_fraud).sum())
        fp = flagged_count - fraud_captured
        rows.append({
            'window_steps': window,
            'flagged_count': flagged_count,
            'flagged_pct': flagged_count / n if n else float('nan'),
            'fraud_captured': fraud_captured,
            'precision': fraud_captured / flagged_count if flagged_count else float('nan'),
            'recall': fraud_captured / total_frauds if total_frauds else float('nan'),
            'false_positive_count': fp,
            'false_positive_rate': fp / total_legitimate if total_legitimate else float('nan'),
        })
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()

    subset, rows_processed, total_fraud_all_types = load_transfer_cashout_subset(DATASET_PATH)

    sender_counts, sender_medians = compute_sender_aggregates(subset)
    rapid_gap = compute_rapid_cashout_gap(subset)
    flags, thresholds = compute_flags(subset, sender_counts, sender_medians, rapid_gap, RAPID_CASHOUT_DEFAULT_WINDOW)

    is_fraud = subset['isFraud'].to_numpy().astype(bool)
    total_frauds = int(is_fraud.sum())
    total_legitimate = len(subset) - total_frauds

    results_df, fraud_flags = compute_metrics(flags, is_fraud, total_frauds, total_legitimate)
    overlap_matrix, unique_counts = compute_overlap(fraud_flags)
    rapid_sensitivity = evaluate_rapid_cashout_windows(
        rapid_gap, is_fraud, RAPID_CASHOUT_WINDOWS, total_frauds, total_legitimate
    )

    elapsed = time.perf_counter() - start

    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    print('=' * 100)
    print('AML SIGNAL EVALUATION -- PaySim ground truth (isFraud)')
    print('=' * 100)

    print(
        '\nAll precision/recall/false-positive-rate figures below are computed over the '
        'analyzed TRANSFER/CASH_OUT subset (not the full 6-type dataset). See the fraud-count '
        'cross-check in Summary statistics.'
    )

    print('\nThresholds computed from the data (over the TRANSFER/CASH_OUT population):')
    print(f"  large_amount        >= {thresholds['large_amount_threshold']:.2f}  (99th percentile of amount)")
    print(f"  rapid_cash_out      <= {RAPID_CASHOUT_DEFAULT_WINDOW} step gap for the main table; full sweep over {RAPID_CASHOUT_WINDOWS} reported below")
    print(f"  velocity            >  {thresholds['velocity_threshold']:.2f}  (95th percentile of sender txn count)")
    print(f"  relative_deviation  >= {thresholds['relative_deviation_threshold']:.4f}x sender median (95th percentile of ratio, senders with >={RELATIVE_DEVIATION_MIN_SENDER_TXNS} txns)")
    print(f"  structuring_strict  >= {STRUCTURING_STRICT_MIN_COUNT} txns in [{STRUCTURING_STRICT_BAND[0]:.0f}, {STRUCTURING_STRICT_BAND[1]:.0f}]  (separate signal from relaxed)")
    print(f"  structuring_relaxed >= {STRUCTURING_RELAXED_MIN_COUNT} txns in [{STRUCTURING_RELAXED_BAND[0]:.0f}, {STRUCTURING_RELAXED_BAND[1]:.0f}]  (separate signal from strict)")

    print('\nSignal comparison (sorted by recall desc, then precision desc):')
    print(results_df.to_string(index=False))
    zero_flag_signals = results_df.loc[results_df['flagged_count'] == 0, 'signal'].tolist()
    if zero_flag_signals:
        print(
            f"\n  NOTE: {zero_flag_signals} produced zero flagged transactions on this dataset. "
            "This is reported as a genuine finding about the signal/dataset, not a script error."
        )

    print('\nRapid cash-out sensitivity across step-gap windows (same precomputed gap array for every row):')
    print(rapid_sensitivity.to_string(index=False))

    print('\nPairwise overlap matrix -- frauds caught by BOTH signal[row] and signal[col]:')
    print(overlap_matrix.to_string())
    print('(diagonal = total frauds caught by that signal alone, i.e. fraud_captured)')

    print('\nFrauds caught ONLY by this signal (no other signal in the set caught them):')
    for name in SIGNAL_NAMES:
        print(f'  {name:22s} {unique_counts[name]}')

    print('\nSummary statistics:')
    print(f'  rows_processed (full CSV, all 6 types)      = {rows_processed:,}')
    print(f'  TRANSFER/CASH_OUT rows analyzed             = {len(subset):,}')
    print(f'  total_frauds (full dataset, all 6 types)    = {total_fraud_all_types:,}')
    print(f'  total_frauds (TRANSFER/CASH_OUT subset)     = {total_frauds:,}')
    if total_fraud_all_types == total_frauds:
        print('  CROSS-CHECK PASSED: every fraud row in the full dataset falls inside the TRANSFER/CASH_OUT subset.')
    else:
        print(
            f'  CROSS-CHECK FAILED: {total_fraud_all_types - total_frauds} fraud row(s) exist outside '
            'TRANSFER/CASH_OUT -- recall above is scoped to the subset only and understates true recall.'
        )
    print(f'  total_legitimate (TRANSFER/CASH_OUT subset) = {total_legitimate:,}')
    print(f'  runtime_seconds                              = {elapsed:.2f}')


if __name__ == '__main__':
    main()
