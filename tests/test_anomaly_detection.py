"""
Unit tests for ML anomaly detection algorithms and XAI explainability engine.
"""

import pytest
import pandas as pd
import numpy as np
from src.anomaly_detection import KeystrokeAnomalyDetector, explain_keystroke_anomaly
from src.feature_extraction import extract_session_features, get_feature_matrix
from src.data_generator import generate_synthetic_dataset
from src.preprocessing import pair_keystroke_events


@pytest.fixture
def sample_feature_data():
    df_raw = generate_synthetic_dataset(num_sessions=30, anomaly_ratio=0.30, random_state=42)
    df_paired = pair_keystroke_events(df_raw)
    df_features = extract_session_features(df_paired)
    return df_features


def test_isolation_forest_fit_predict(sample_feature_data):
    X, y = get_feature_matrix(sample_feature_data)
    detector = KeystrokeAnomalyDetector(algorithm="Isolation Forest", contamination=0.20, random_state=42)
    detector.fit(X)

    preds = detector.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    scores = detector.compute_anomaly_scores(X)
    assert len(scores) == len(X)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_one_class_svm(sample_feature_data):
    X, _ = get_feature_matrix(sample_feature_data)
    detector = KeystrokeAnomalyDetector(algorithm="One-Class SVM", contamination=0.15)
    detector.fit(X)
    preds = detector.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_local_outlier_factor(sample_feature_data):
    X, _ = get_feature_matrix(sample_feature_data)
    detector = KeystrokeAnomalyDetector(algorithm="Local Outlier Factor", contamination=0.15)
    detector.fit(X)
    preds = detector.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_model_evaluation(sample_feature_data):
    X, y = get_feature_matrix(sample_feature_data)
    detector = KeystrokeAnomalyDetector(algorithm="Isolation Forest", contamination=0.30, random_state=42)
    detector.fit(X)
    metrics = detector.evaluate(X, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] >= 0.0


def test_explain_keystroke_anomaly_bot():
    # Simulated bot row: very low CV, superhuman WPM, tiny dwell
    bot_row = pd.Series({
        "words_per_min": 180.0,
        "keys_per_sec": 15.0,
        "flight_cv": 0.02,
        "dwell_mean": 15.0,
        "flight_mean": 12.0,
        "error_rate": 0.0
    })
    explanations = explain_keystroke_anomaly(bot_row)
    severities = [exp["severity"] for exp in explanations]
    assert "CRITICAL" in severities


def test_explain_keystroke_anomaly_human():
    human_row = pd.Series({
        "words_per_min": 52.0,
        "keys_per_sec": 4.5,
        "flight_cv": 0.42,
        "dwell_mean": 110.0,
        "flight_mean": 175.0,
        "error_rate": 0.03
    })
    explanations = explain_keystroke_anomaly(human_row)
    assert explanations[0]["severity"] == "SAFE"
