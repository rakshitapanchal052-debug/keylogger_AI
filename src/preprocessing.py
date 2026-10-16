"""
Data Preprocessing Module for Keystroke Behavior Analysis.
Validates raw keystroke streams, pairs key-down and key-up events,
computes dwell and flight times, filters outliers, and ensures privacy protection.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np

REQUIRED_EVENT_COLUMNS = {"session_id", "key", "event_type", "timestamp"}


def validate_event_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the DataFrame has the minimum required columns and proper data types.
    """
    if df is None or df.empty:
        return False, "Provided dataset is empty or None."

    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    # Ensure timestamp is numeric
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        try:
            pd.to_numeric(df["timestamp"])
        except Exception:
            return False, "Column 'timestamp' must contain numeric millisecond values."

    return True, "Dataset structure is valid."


def pair_keystroke_events(
    df_events: pd.DataFrame,
    anonymize_keys: bool = False,
    max_dwell_ms: float = 3000.0,
    max_flight_ms: float = 5000.0
) -> pd.DataFrame:
    """
    Pair sequential press and release events for each key within a session.
    Computes:
      - dwell_time_ms = release_time - press_time
      - flight_time_ms = current_press_time - previous_release_time
    """
    valid, msg = validate_event_data(df_events)
    if not valid:
        raise ValueError(f"Invalid dataset: {msg}")

    # Standardize column types
    df = df_events.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df["key"] = df["key"].astype(str)

    if "user_id" not in df.columns:
        df["user_id"] = "USER_DEFAULT"
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0
    if "profile" not in df.columns:
        df["profile"] = "unknown"

    # Sort chronologically by session and timestamp
    df = df.sort_values(by=["session_id", "timestamp"]).reset_index(drop=True)

    paired_records = []

    for session_id, group in df.groupby("session_id", sort=False):
        user_id = group["user_id"].iloc[0]
        profile = group["profile"].iloc[0]
        is_anomaly = group["is_anomaly"].iloc[0]

        # Stack for unclosed press events per key
        open_presses = {}
        prev_release_time: Optional[float] = None

        for _, row in group.iterrows():
            k = row["key"]
            etype = row["event_type"]
            t = row["timestamp"]

            if etype in ["press", "keydown", "down"]:
                if k not in open_presses:
                    open_presses[k] = []
                open_presses[k].append(t)

            elif etype in ["release", "keyup", "up"]:
                if k in open_presses and len(open_presses[k]) > 0:
                    press_time = open_presses[k].pop(0)
                    dwell = t - press_time

                    if 0.0 <= dwell <= max_dwell_ms:
                        # Flight time from previous key release
                        if prev_release_time is not None:
                            flight = press_time - prev_release_time
                            # Bound flight time for pauses
                            flight = max(0.0, min(flight, max_flight_ms))
                        else:
                            flight = 0.0

                        display_key = "[KEY]" if (anonymize_keys and len(k) == 1 and k.isalnum()) else k

                        paired_records.append({
                            "session_id": session_id,
                            "user_id": user_id,
                            "profile": profile,
                            "key": display_key,
                            "press_time": press_time,
                            "release_time": t,
                            "dwell_time_ms": round(dwell, 2),
                            "flight_time_ms": round(flight, 2),
                            "is_anomaly": is_anomaly
                        })

                        prev_release_time = t

    return pd.DataFrame(paired_records)


def filter_timing_outliers(
    df_paired: pd.DataFrame,
    min_dwell: float = 1.0,
    max_dwell: float = 2000.0,
    max_flight: float = 4000.0
) -> pd.DataFrame:
    """
    Remove extreme artifacts or broken sensor readings while preserving bot signatures.
    """
    if df_paired.empty:
        return df_paired

    condition = (
        (df_paired["dwell_time_ms"] >= min_dwell) &
        (df_paired["dwell_time_ms"] <= max_dwell) &
        (df_paired["flight_time_ms"] <= max_flight)
    )
    return df_paired[condition].reset_index(drop=True)
