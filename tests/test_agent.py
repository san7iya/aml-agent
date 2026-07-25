import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sample_dataset(tmp_path):
    data_path = tmp_path / 'sample.csv'
    pd.DataFrame(
        {
            'step': [1, 2, 3, 4, 5],
            'type': ['TRANSFER', 'TRANSFER', 'TRANSFER', 'CASH_OUT', 'PAYMENT'],
            'amount': [9500.0, 9600.0, 9800.0, 150000.0, 100.0],
            'nameOrig': ['C1231006815', 'C1231006815', 'C1231006815', 'C2222222222', 'C3333333333'],
            'oldbalanceOrg': [100000.0, 90000.0, 80400.0, 250000.0, 300.0],
            'newbalanceOrig': [90000.0, 80400.0, 70600.0, 100000.0, 200.0],
            'nameDest': ['M1', 'M2', 'M3', 'M4', 'M5'],
            'oldbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
            'newbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
            'isFraud': [0, 0, 0, 1, 0],
            'isFlaggedFraud': [0, 0, 0, 0, 0],
        }
    ).to_csv(data_path, index=False)
    return data_path


def test_intent_parsing_and_routing():
    agent = load_module('aml_agent', 'aml_agent.py')

    query = 'Find structuring patterns in the last 30 days'
    plan = agent.build_plan(query)
    assert plan['intent'] == 'structuring'
    assert 'feature_engineering' in plan['tools']
    assert 'anomaly_detection' in plan['tools']

    query = 'Is customer ID C1231006815 suspicious?'
    plan = agent.build_plan(query)
    assert plan['intent'] == 'customer_risk'
    assert 'customer_profile' in plan['tools']

    query = 'Show me high-risk transactions from the past week'
    plan = agent.build_plan(query)
    assert plan['intent'] == 'transaction_risk'
    assert 'risk_scoring' in plan['tools']


def test_aggregation_phrasing_without_entity_id_routes_to_transaction_risk(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = build_sample_dataset(tmp_path)

    query = 'which customers made 10+ transactions under $10,000'
    plan = agent.build_plan(query)
    assert plan['intent'] == 'transaction_risk'
    assert plan['entity_id'] is None

    result = agent.analyze_query(query, data_path)
    assert result['plan']['intent'] == 'transaction_risk'
    assert 'flagged_transactions' in result['result']
    assert result['result']['flagged_transactions']
    assert result['result']['reason'] != 'No specific pattern was detected from the query terms.'


def test_analyze_query_dispatches_real_intent_logic(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = build_sample_dataset(tmp_path)

    structuring_result = agent.analyze_query('Find structuring patterns in the last 30 days', data_path)
    transaction_result = agent.analyze_query('Show me high-risk transactions from the past week', data_path)
    customer_result = agent.analyze_query('Is customer ID C1231006815 suspicious?', data_path)

    assert structuring_result['plan']['intent'] == 'structuring'
    assert structuring_result['result']['matched_senders']
    assert structuring_result['result']['risk_band'] in {'medium', 'high'}

    assert transaction_result['plan']['intent'] == 'transaction_risk'
    assert transaction_result['result']['flagged_transactions']
    assert transaction_result['result']['risk_band'] in {'medium', 'high'}

    assert customer_result['plan']['intent'] == 'customer_risk'
    assert customer_result['result']['entity_id'] == 'C1231006815'
    assert customer_result['result']['risk_band'] in {'medium', 'high'}

    assert structuring_result['result']['reason'] != transaction_result['result']['reason']
    assert structuring_result['result']['reason'] != customer_result['result']['reason']


def test_transaction_risk_exposes_flagged_count_metadata(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = build_sample_dataset(tmp_path)

    transaction_result = agent.analyze_query('Show me high-risk transactions from the past week', data_path)

    assert 'flagged_count' in transaction_result['result']
    assert 'flagged_rate' in transaction_result['result']
    assert transaction_result['result']['flagged_count'] >= len(transaction_result['result']['flagged_transactions'])


def test_transaction_risk_reason_describes_composite_signals(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = build_sample_dataset(tmp_path)

    transaction_result = agent.analyze_query('Show me high-risk transactions from the past week', data_path)

    assert 'composite' in transaction_result['result']['reason'].lower()
    assert 'velocity' in transaction_result['result']['reason'].lower() or 'structuring' in transaction_result['result']['reason'].lower()


def test_transaction_risk_flags_behavioral_signals_over_amount_artifact(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = tmp_path / 'behavioral.csv'
    pd.DataFrame(
        {
            'step': [1, 2, 3, 10],
            'type': ['TRANSFER', 'TRANSFER', 'TRANSFER', 'CASH_OUT'],
            'amount': [9500.0, 9600.0, 9800.0, 5000.0],
            'nameOrig': ['C1231006815', 'C1231006815', 'C1231006815', 'C1231006815'],
            'oldbalanceOrg': [100000.0, 90000.0, 80400.0, 60000.0],
            'newbalanceOrig': [90000.0, 80400.0, 70600.0, 55000.0],
            'nameDest': ['C1', 'C2', 'C3', 'C4'],
            'oldbalanceDest': [0.0, 0.0, 0.0, 0.0],
            'newbalanceDest': [0.0, 0.0, 0.0, 0.0],
            'isFraud': [0, 0, 0, 0],
            'isFlaggedFraud': [0, 0, 0, 0],
        }
    ).to_csv(data_path, index=False)

    transaction_result = agent.analyze_query('Show me high-risk transactions from the past week', data_path)

    flagged_rows = transaction_result['result']['flagged_transactions']
    assert flagged_rows
    assert any(item['amount'] != 10000000.0 for item in flagged_rows)
    assert any(item['signals']['structuring'] for item in flagged_rows)


def test_transaction_risk_ignores_merchant_destination_mismatch(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = tmp_path / 'merchant_mismatch.csv'
    pd.DataFrame(
        {
            'step': [1, 2],
            'type': ['TRANSFER', 'TRANSFER'],
            'amount': [10.0, 1000.0],
            'nameOrig': ['C1231006815', 'C1231006815'],
            'oldbalanceOrg': [100.0, 2000.0],
            'newbalanceOrig': [90.0, 1000.0],
            'nameDest': ['M1', 'C2'],
            'oldbalanceDest': [999.0, 0.0],
            'newbalanceDest': [1000.0, 0.0],
            'isFraud': [0, 0],
            'isFlaggedFraud': [0, 0],
        }
    ).to_csv(data_path, index=False)

    transaction_result = agent.analyze_query('Show me high-risk transactions from the past week', data_path)

    assert transaction_result['result']['flagged_count'] == 0
    assert transaction_result['result']['flagged_rate'] == 0.0
    assert transaction_result['result']['flagged_transactions'] == []


def test_eda_query_routes_to_eda_intent_and_returns_expected_structure(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')
    data_path = build_sample_dataset(tmp_path)

    plan = agent.build_plan('Give me a summary of this dataset')
    assert plan['intent'] == 'eda'

    eda_result = agent.analyze_query('Profile this dataset', data_path)

    assert eda_result['plan']['intent'] == 'eda'
    result = eda_result['result']
    assert result['total_rows'] == 5
    assert result['fraud_count'] == 1
    assert 'fraud_rate' in result
    assert 'type_breakdown' in result
    assert 'TRANSFER_CASH_OUT' in result['type_breakdown']
    assert 'other' in result['type_breakdown']
    assert 'signal_validation_summary' in result
    assert 'large_amount' in result['validated_signals']
    assert 'risk_band' in result


def test_build_working_subset_accepts_custom_data_path(tmp_path):
    agent = load_module('aml_agent', 'aml_agent.py')

    data_path = tmp_path / 'sample.csv'
    pd.DataFrame(
        {
            'step': [1, 2],
            'type': ['PAYMENT', 'TRANSFER'],
            'amount': [100.0, 500.0],
            'nameOrig': ['C1', 'C2'],
            'oldbalanceOrg': [100.0, 500.0],
            'newbalanceOrig': [0.0, 0.0],
            'nameDest': ['M1', 'M2'],
            'oldbalanceDest': [0.0, 0.0],
            'newbalanceDest': [0.0, 0.0],
            'isFraud': [0, 1],
            'isFlaggedFraud': [0, 0],
        }
    ).to_csv(data_path, index=False)

    output_path = tmp_path / 'subset.csv'
    built_path = agent.build_working_subset(output_path=output_path, sample_size=1, data_path=data_path)

    subset = pd.read_csv(built_path)
    assert built_path.exists()
    assert len(subset) == 2
    assert subset['isFraud'].sum() == 1