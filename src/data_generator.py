"""
Synthetic Keystroke Dataset Generator.
Generates realistic event-level and session-level keystroke timing datasets
simulating both legitimate human typists and anomalous bot/script activities.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

SAMPLE_PHRASES = [
    "the quick brown fox jumps over the lazy dog",
    "cybersecurity analysis prevents unauthorized system intrusion",
    "biometric keystroke dynamics provide continuous authentication",
    "anomaly detection algorithms identify automated credential stuffing attacks",
    "machine learning models analyze typing flight time and dwell duration"
]

PROFILE_SPECS = {
    "human_regular": {
        "dwell_mean": 105.0,
        "dwell_std": 22.0,
        "flight_mean": 180.0,
        "flight_std": 55.0,
        "error_rate": 0.04,
        "is_anomaly": 0,
        "description": "Standard Human Typist (natural rhythm, moderate variance)"
    },
    "human_expert": {
        "dwell_mean": 70.0,
        "dwell_std": 14.0,
        "flight_mean": 110.0,
        "flight_std": 28.0,
        "error_rate": 0.02,
        "is_anomaly": 0,
        "description": "Touch Typist (fast cadence, fluid transitions)"
    },
    "human_hesitant": {
        "dwell_mean": 150.0,
        "dwell_std": 45.0,
        "flight_mean": 380.0,
        "flight_std": 140.0,
        "error_rate": 0.08,
        "is_anomaly": 0,
        "description": "Hesitant/Novice Typist (higher dwell, frequent pauses)"
    },
    "bot_macro_replay": {
        "dwell_mean": 18.0,
        "dwell_std": 0.8,
        "flight_mean": 20.0,
        "flight_std": 1.2,
        "error_rate": 0.0,
        "is_anomaly": 1,
        "description": "Automated Macro Replay (rigid periodicity, near-zero variance)"
    },
    "bot_credential_stuffing": {
        "dwell_mean": 8.0,
        "dwell_std": 0.4,
        "flight_mean": 9.0,
        "flight_std": 0.6,
        "error_rate": 0.0,
        "is_anomaly": 1,
        "description": "Credential Stuffing Bot (rapid mechanical burst)"
    },
    "bot_paste_injection": {
        "dwell_mean": 2.5,
        "dwell_std": 0.3,
        "flight_mean": 2.0,
        "flight_std": 0.4,
        "error_rate": 0.0,
        "is_anomaly": 1,
        "description": "Programmatic Paste/Injection (instantaneous key events)"
    }
}


def generate_session_events(
    session_id: str,
    user_id: str,
    profile_name: str,
    phrase: str,
    start_time: float = 0.0,
    random_state: int = None
) -> Tuple[pd.DataFrame, float]:
    """
    Generate chronological key press and release events for a specific phrase and profile.
    Returns:
        tuple (DataFrame of events, session_end_time)
    """
    rng = np.random.default_rng(random_state)
    spec = PROFILE_SPECS.get(profile_name, PROFILE_SPECS["human_regular"])

    events: List[Dict[str, Any]] = []
    current_time = start_time

    chars = list(phrase)
    i = 0
    while i < len(chars):
        char = chars[i]

        # Check if an error/backspace should be simulated (human profiles only)
        simulate_error = (spec["error_rate"] > 0) and (rng.random() < spec["error_rate"])

        if simulate_error:
            # Type a typo key, then backspace
            typo_key = rng.choice(["x", "z", "q", "k", "p"])
            typo_dwell = max(5.0, rng.normal(spec["dwell_mean"], spec["dwell_std"]))
            typo_press = current_time
            typo_release = typo_press + typo_dwell

            events.append({
                "session_id": session_id,
                "user_id": user_id,
                "profile": profile_name,
                "key": typo_key,
                "event_type": "press",
                "timestamp": round(typo_press, 2),
                "is_anomaly": spec["is_anomaly"]
            })
            events.append({
                "session_id": session_id,
                "user_id": user_id,
                "profile": profile_name,
                "key": typo_key,
                "event_type": "release",
                "timestamp": round(typo_release, 2),
                "is_anomaly": spec["is_anomaly"]
            })

            # Reaction time before backspace
            reaction_flight = max(10.0, rng.normal(spec["flight_mean"] * 1.3, spec["flight_std"]))
            bs_press = typo_release + reaction_flight
            bs_dwell = max(5.0, rng.normal(spec["dwell_mean"], spec["dwell_std"]))
            bs_release = bs_press + bs_dwell

            events.append({
                "session_id": session_id,
                "user_id": user_id,
                "profile": profile_name,
                "key": "Backspace",
                "event_type": "press",
                "timestamp": round(bs_press, 2),
                "is_anomaly": spec["is_anomaly"]
            })
            events.append({
                "session_id": session_id,
                "user_id": user_id,
                "profile": profile_name,
                "key": "Backspace",
                "event_type": "release",
                "timestamp": round(bs_release, 2),
                "is_anomaly": spec["is_anomaly"]
            })

            current_time = bs_release

        # Flight time before typing the target character
        flight = max(1.0, rng.normal(spec["flight_mean"], spec["flight_std"]))
        press_time = current_time + flight
        dwell = max(2.0, rng.normal(spec["dwell_mean"], spec["dwell_std"]))
        release_time = press_time + dwell

        events.append({
            "session_id": session_id,
            "user_id": user_id,
            "profile": profile_name,
            "key": char,
            "event_type": "press",
            "timestamp": round(press_time, 2),
            "is_anomaly": spec["is_anomaly"]
        })
        events.append({
            "session_id": session_id,
            "user_id": user_id,
            "profile": profile_name,
            "key": char,
            "event_type": "release",
            "timestamp": round(release_time, 2),
            "is_anomaly": spec["is_anomaly"]
        })

        current_time = release_time
        i += 1

    df_events = pd.DataFrame(events)
    return df_events, current_time


def generate_synthetic_dataset(
    num_sessions: int = 120,
    anomaly_ratio: float = 0.20,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a full synthetic dataset consisting of multiple typing sessions across
    human profiles (normal) and bot/macro profiles (anomalous).
    """
    rng = np.random.default_rng(random_state)
    all_events: List[pd.DataFrame] = []

    human_profiles = ["human_regular", "human_expert", "human_hesitant"]
    human_weights = [0.60, 0.25, 0.15]

    bot_profiles = ["bot_macro_replay", "bot_credential_stuffing", "bot_paste_injection"]
    bot_weights = [0.45, 0.35, 0.20]

    for s_idx in range(num_sessions):
        session_id = f"SESS_{s_idx+1:04d}"
        is_anomaly = rng.random() < anomaly_ratio

        if is_anomaly:
            profile = rng.choice(bot_profiles, p=bot_weights)
            user_id = f"BOT_{rng.integers(1, 10):02d}"
        else:
            profile = rng.choice(human_profiles, p=human_weights)
            user_id = f"USER_{rng.integers(1, 15):02d}"

        phrase = rng.choice(SAMPLE_PHRASES)
        seed = int(rng.integers(0, 100000))
        df_sess, _ = generate_session_events(
            session_id=session_id,
            user_id=user_id,
            profile_name=profile,
            phrase=phrase,
            start_time=0.0,
            random_state=seed
        )
        all_events.append(df_sess)

    combined_df = pd.concat(all_events, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    df = generate_synthetic_dataset(num_sessions=60, anomaly_ratio=0.25, random_state=42)
    print(f"Generated {len(df)} keystroke events across {df['session_id'].nunique()} sessions.")
    print(df.head(10))
    df.to_csv("data/sample_keystrokes.csv", index=False)
    print("Saved sample dataset to data/sample_keystrokes.csv")
