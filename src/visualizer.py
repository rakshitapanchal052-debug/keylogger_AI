"""
Visualization Module for Keylogger_AI Keystroke Behavior Analysis.
Generates responsive Plotly charts adhering to a dark navy and cyber neon aesthetic.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Cybersecurity Dark Palette
THEME = {
    "paper_bg": "#0B132B",
    "plot_bg": "#111D3B",
    "grid_color": "#1C2B54",
    "text_color": "#E0E6ED",
    "color_normal": "#00FF87",   # Cyber Green
    "color_anomaly": "#FF0055",  # Crimson Neon
    "color_accent": "#00E5FF",   # Cyan
    "color_warning": "#FFB703",  # Amber
    "font_family": "Segoe UI, sans-serif"
}


def _apply_cyber_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent cyber dark navy styling to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=THEME["text_color"], size=16, family=THEME["font_family"])),
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text_color"], family=THEME["font_family"]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(11, 19, 43, 0.7)",
            bordercolor=THEME["grid_color"],
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            color=THEME["text_color"]
        ),
        yaxis=dict(
            gridcolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            color=THEME["text_color"]
        )
    )
    return fig


def plot_dwell_vs_flight(df_features: pd.DataFrame, pred_col: str = "is_anomaly") -> go.Figure:
    """
    Scatter plot comparing mean dwell time vs mean flight time with anomaly coloring.
    """
    df = df_features.copy()
    if pred_col not in df.columns:
        df[pred_col] = 0

    df["Status"] = df[pred_col].apply(lambda x: "Malicious / Anomaly" if x == 1 else "Normal Typist")

    color_map = {
        "Normal Typist": THEME["color_normal"],
        "Malicious / Anomaly": THEME["color_anomaly"]
    }

    fig = px.scatter(
        df,
        x="flight_mean",
        y="dwell_mean",
        color="Status",
        size="words_per_min",
        hover_data=["session_id", "user_id", "keys_per_sec", "flight_cv"],
        color_discrete_map=color_map,
        labels={
            "flight_mean": "Mean Flight Time (ms)",
            "dwell_mean": "Mean Dwell Time (ms)",
            "words_per_min": "Typing Speed (WPM)"
        },
        title="<b>Keystroke Dynamics: Flight Time vs Dwell Time</b>"
    )
    return _apply_cyber_layout(fig)


def plot_speed_vs_regularity(df_features: pd.DataFrame, pred_col: str = "is_anomaly") -> go.Figure:
    """
    Scatter plot of typing speed vs timing regularity (CV).
    Automated scripts cluster at the bottom-right (high speed, near-zero variance).
    """
    df = df_features.copy()
    if pred_col not in df.columns:
        df[pred_col] = 0

    df["Status"] = df[pred_col].apply(lambda x: "Malicious / Anomaly" if x == 1 else "Normal Typist")
    color_map = {
        "Normal Typist": THEME["color_normal"],
        "Malicious / Anomaly": THEME["color_anomaly"]
    }

    fig = px.scatter(
        df,
        x="words_per_min",
        y="flight_cv",
        color="Status",
        hover_data=["session_id", "user_id", "profile"],
        color_discrete_map=color_map,
        labels={
            "words_per_min": "Typing Velocity (WPM)",
            "flight_cv": "Rhythm Regularity (Flight CV)"
        },
        title="<b>Bot Detection Boundary: Velocity vs Rhythm Regularity</b>"
    )

    # Highlight anomalous bot quadrant
    fig.add_hrect(
        y0=0.0, y1=0.08,
        fillcolor="rgba(255, 0, 85, 0.12)",
        line_width=1,
        line_dash="dot",
        line_color=THEME["color_anomaly"],
        annotation_text="Script / Bot Zone (CV < 0.08)",
        annotation_position="top right"
    )

    return _apply_cyber_layout(fig)


def plot_timing_distributions(df_paired: pd.DataFrame) -> go.Figure:
    """
    Histogram distributions of Dwell and Flight times.
    """
    df = df_paired.copy()
    df["Status"] = df["is_anomaly"].apply(lambda x: "Anomaly" if x == 1 else "Normal")

    fig = go.Figure()

    # Dwell distributions
    fig.add_trace(go.Histogram(
        x=df[df["is_anomaly"] == 0]["dwell_time_ms"],
        name="Normal Dwell Time",
        marker_color=THEME["color_normal"],
        opacity=0.6,
        nbinsx=40
    ))
    fig.add_trace(go.Histogram(
        x=df[df["is_anomaly"] == 1]["dwell_time_ms"],
        name="Anomaly Dwell Time",
        marker_color=THEME["color_anomaly"],
        opacity=0.6,
        nbinsx=40
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Time (ms)",
        yaxis_title="Keystroke Event Count",
        title="<b>Distribution of Key Dwell Times (Hold Duration)</b>"
    )
    return _apply_cyber_layout(fig)


def plot_biometric_radar(
    sample_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    title: str = "Biometric Profile Comparison"
) -> go.Figure:
    """
    Radar chart comparing a candidate sample against baseline human profile.
    """
    categories = [
        "Typing Speed (WPM)",
        "Dwell Duration",
        "Flight Latency",
        "Rhythm Regularity",
        "Error / Backspaces",
        "Key Consistency"
    ]

    # Normalize metrics to 0-100 scale for intuitive radar comparison
    def normalize_metrics(m: Dict[str, float]) -> List[float]:
        return [
            min(100.0, (m.get("words_per_min", 45) / 120.0) * 100),
            min(100.0, (m.get("dwell_mean", 100) / 200.0) * 100),
            min(100.0, (m.get("flight_mean", 180) / 400.0) * 100),
            min(100.0, max(0.0, 100 - (m.get("flight_cv", 0.4) * 150))),
            min(100.0, (m.get("error_rate", 0.04) / 0.15) * 100),
            min(100.0, max(0.0, 100 - (m.get("dwell_cv", 0.3) * 200)))
        ]

    val_sample = normalize_metrics(sample_metrics)
    val_baseline = normalize_metrics(baseline_metrics)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=val_baseline + [val_baseline[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Legitimate Baseline",
        line=dict(color=THEME["color_normal"], width=2),
        fillcolor="rgba(0, 255, 135, 0.2)"
    ))

    fig.add_trace(go.Scatterpolar(
        r=val_sample + [val_sample[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Candidate Sample",
        line=dict(color=THEME["color_anomaly"], width=2, dash="dash"),
        fillcolor="rgba(255, 0, 85, 0.25)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=THEME["text_color"], gridcolor=THEME["grid_color"]),
            angularaxis=dict(color=THEME["text_color"], gridcolor=THEME["grid_color"]),
            bgcolor=THEME["plot_bg"]
        ),
        title=f"<b>{title}</b>"
    )
    return _apply_cyber_layout(fig)


def plot_risk_gauge(risk_score: float) -> go.Figure:
    """
    Speedometer gauge showing Anomaly Risk (0% to 100%).
    """
    score_pct = round(risk_score * 100.0, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pct,
        number={"suffix": "%", "font": {"color": THEME["text_color"], "size": 36, "family": THEME["font_family"]}},
        title={"text": "<b>Keystroke Anomaly Risk Score</b>", "font": {"color": THEME["text_color"], "size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": THEME["text_color"]},
            "bar": {"color": THEME["color_anomaly"] if score_pct >= 50 else THEME["color_normal"]},
            "bgcolor": THEME["plot_bg"],
            "borderwidth": 1,
            "bordercolor": THEME["grid_color"],
            "steps": [
                {"range": [0, 35], "color": "rgba(0, 255, 135, 0.25)"},
                {"range": [35, 70], "color": "rgba(255, 183, 3, 0.25)"},
                {"range": [70, 100], "color": "rgba(255, 0, 85, 0.35)"}
            ],
            "threshold": {
                "line": {"color": THEME["color_anomaly"], "width": 4},
                "thickness": 0.85,
                "value": 70
            }
        }
    ))
    return _apply_cyber_layout(fig)


def plot_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """
    Plot confusion matrix as a heatmap.
    """
    labels = ["Normal (0)", "Anomaly (1)"]
    z = cm
    z_text = [[str(val) for val in row] for row in z]

    fig = ff_heatmap = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "#FFFFFF"},
        colorscale=[[0, "#111D3B"], [1, "#00E5FF"]],
        showscale=False
    ))

    fig.update_layout(
        title="<b>Detection Confusion Matrix</b>",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    return _apply_cyber_layout(fig)
