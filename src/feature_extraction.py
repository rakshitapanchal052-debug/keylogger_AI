"""
Feature Extraction Module for Keystroke Biometrics and Anomaly Detection.
Aggregates event-level keystroke timing into rich multi-dimensional behavioral feature vectors.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


COMMON_DIGRAPHS = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"]


def extract_session_features(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    Extract aggregated statistical and behavioral features for each session in df_paired.
    """
    if df_paired.empty:
        return pd.DataFrame()

    features_list: List[Dict[str, Any]] = []

    for session_id, group in df_paired.groupby("session_id", sort=False):
        user_id = group["user_id"].iloc[0] if "user_id" in group.columns else "UNKNOWN"
        profile = group["profile"].iloc[0] if "profile" in group.columns else "unknown"
        is_anomaly = int(group["is_anomaly"].iloc[0]) if "is_anomaly" in group.columns else 0

        keystrokes = len(group)
        if keystrokes < 3:
            continue

        dwell_vals = group["dwell_time_ms"].values
        # Exclude initial 0 flight time
        flight_vals = group["flight_time_ms"].iloc[1:].values if len(group) > 1 else np.array([0.0])

        if len(flight_vals) == 0:
            flight_vals = np.array([0.0])

        session_start = group["press_time"].min()
        session_end = group["release_time"].max()
        duration_sec = max(0.001, (session_end - session_start) / 1000.0)

        # Typing Speed Metrics
        kps = keystrokes / duration_sec
        wpm = (kps * 60.0) / 5.0

        # Dwell Time Statistics
        dwell_mean = float(np.mean(dwell_vals))
        dwell_std = float(np.std(dwell_vals, ddof=1)) if len(dwell_vals) > 1 else 0.0
        dwell_median = float(np.median(dwell_vals))
        dwell_q25 = float(np.percentile(dwell_vals, 25))
        dwell_q75 = float(np.percentile(dwell_vals, 75))
        dwell_iqr = dwell_q75 - dwell_q25
        dwell_cv = dwell_std / (dwell_mean + 1e-6)

        # Flight Time Statistics
        flight_mean = float(np.mean(flight_vals))
        flight_std = float(np.std(flight_vals, ddof=1)) if len(flight_vals) > 1 else 0.0
        flight_median = float(np.median(flight_vals))
        flight_q25 = float(np.percentile(flight_vals, 25))
        flight_q75 = float(np.percentile(flight_vals, 75))
        flight_iqr = flight_q75 - flight_q25
        flight_cv = flight_std / (flight_mean + 1e-6)

        # Ratio & Rhythm
        dwell_flight_ratio = dwell_mean / (flight_mean + 1e-6)

        # Error / Backspace Analysis
        keys_str = group["key"].astype(str).str.lower().tolist()
        backspace_count = sum(1 for k in keys_str if "backspace" in k or k == "delete")
        error_rate = backspace_count / keystrokes

        # Pauses (flight > 1000ms)
        pause_count = int(np.sum(flight_vals > 1000.0))
        pause_ratio = pause_count / max(1, len(flight_vals))

        # Digraph Timing (Mean latency between consecutive common digraph characters)
        digraph_latencies = []
        for idx in range(len(keys_str) - 1):
            pair = keys_str[idx] + keys_str[idx + 1]
            if pair in COMMON_DIGRAPHS:
                digraph_latencies.append(flight_vals[min(idx, len(flight_vals) - 1)])

        digraph_latency_mean = float(np.mean(digraph_latencies)) if digraph_latencies else flight_mean

        features_list.append({
            "session_id": session_id,
            "user_id": user_id,
            "profile": profile,
            "keystroke_count": keystrokes,
            "duration_sec": round(duration_sec, 2),
            "keys_per_sec": round(kps, 2),
            "words_per_min": round(wpm, 1),
            "dwell_mean": round(dwell_mean, 2),
            "dwell_std": round(dwell_std, 2),
            "dwell_median": round(dwell_median, 2),
            "dwell_iqr": round(dwell_iqr, 2),
            "dwell_cv": round(dwell_cv, 4),
            "flight_mean": round(flight_mean, 2),
            "flight_std": round(flight_std, 2),
            "flight_median": round(flight_median, 2),
            "flight_iqr": round(flight_iqr, 2),
            "flight_cv": round(flight_cv, 4),
            "dwell_flight_ratio": round(dwell_flight_ratio, 3),
            "error_rate": round(error_rate, 4),
            "backspace_count": backspace_count,
            "pause_count": pause_count,
            "pause_ratio": round(pause_ratio, 3),
            "digraph_latency_mean": round(digraph_latency_mean, 2),
            "is_anomaly": is_anomaly
        })

    return pd.DataFrame(features_list)


FEATURE_COLUMNS = [
    "keys_per_sec",
    "words_per_min",
    "dwell_mean",
    "dwell_std",
    "dwell_median",
    "dwell_iqr",
    "dwell_cv",
    "flight_mean",
    "flight_std",
    "flight_median",
    "flight_iqr",
    "flight_cv",
    "dwell_flight_ratio",
    "error_rate",
    "pause_ratio",
    "digraph_latency_mean"
]


def get_feature_matrix(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract pure numerical feature matrix X and optional target series y from a feature dataframe.
    """
    available_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]
    X = df_features[available_cols].copy()
    y = df_features["is_anomaly"].copy() if "is_anomaly" in df_features.columns else None
    return X, y
