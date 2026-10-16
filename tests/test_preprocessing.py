"""
Unit tests for data preprocessing and event pairing.
"""

import pytest
import pandas as pd
from src.preprocessing import validate_event_data, pair_keystroke_events, filter_timing_outliers
from src.data_generator import generate_session_events


def test_validate_event_data_valid():
    df = pd.DataFrame({
        "session_id": ["S1", "S1"],
        "key": ["a", "a"],
        "event_type": ["press", "release"],
        "timestamp": [100.0, 180.0]
    })
    valid, msg = validate_event_data(df)
    assert valid is True
    assert "valid" in msg.lower()


def test_validate_event_data_missing_column():
    df = pd.DataFrame({
        "session_id": ["S1"],
        "key": ["a"]
    })
    valid, msg = validate_event_data(df)
    assert valid is False
    assert "missing" in msg.lower()


def test_pair_keystroke_events():
    df_raw = pd.DataFrame([
        {"session_id": "S1", "key": "h", "event_type": "press", "timestamp": 100.0},
        {"session_id": "S1", "key": "h", "event_type": "release", "timestamp": 180.0},
        {"session_id": "S1", "key": "i", "event_type": "press", "timestamp": 300.0},
        {"session_id": "S1", "key": "i", "event_type": "release", "timestamp": 390.0},
    ])
    df_paired = pair_keystroke_events(df_raw)

    assert len(df_paired) == 2
    assert "dwell_time_ms" in df_paired.columns
    assert "flight_time_ms" in df_paired.columns

    # First key dwell: 180 - 100 = 80ms
    assert df_paired.iloc[0]["dwell_time_ms"] == 80.0
    # First key flight is 0 (initial)
    assert df_paired.iloc[0]["flight_time_ms"] == 0.0

    # Second key dwell: 390 - 300 = 90ms
    assert df_paired.iloc[1]["dwell_time_ms"] == 90.0
    # Second key flight: 300 - 180 = 120ms
    assert df_paired.iloc[1]["flight_time_ms"] == 120.0


def test_filter_timing_outliers():
    df_paired = pd.DataFrame([
        {"session_id": "S1", "key": "a", "press_time": 0, "release_time": 100, "dwell_time_ms": 100.0, "flight_time_ms": 150.0},
        {"session_id": "S1", "key": "b", "press_time": 250, "release_time": 255, "dwell_time_ms": 0.5, "flight_time_ms": 150.0}, # Outlier < 1ms dwell
        {"session_id": "S1", "key": "c", "press_time": 405, "release_time": 505, "dwell_time_ms": 100.0, "flight_time_ms": 9999.0}, # Outlier flight > 4000ms
    ])
    filtered = filter_timing_outliers(df_paired, min_dwell=1.0, max_flight=4000.0)
    assert len(filtered) == 1
    assert filtered.iloc[0]["key"] == "a"


def test_synthetic_session_generation():
    df_sess, end_time = generate_session_events(
        session_id="TEST_01",
        user_id="USER_01",
        profile_name="human_regular",
        phrase="hello",
        random_state=42
    )
    assert len(df_sess) >= 10  # 5 characters * 2 (press+release) + possible errors
    assert df_sess["session_id"].iloc[0] == "TEST_01"
    assert end_time > 0
