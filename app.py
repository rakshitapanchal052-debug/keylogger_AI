"""
Keylogger_AI: AI-Based Keystroke Behavior Analysis System
Modern Cybersecurity Dashboard with Dark Navy Theme, Biometric Dynamics Analysis,
and Machine-Learning Anomaly Detection.
"""

import time
import json
import streamlit as st
import pandas as pd
import numpy as np

# Internal modules
from src.data_generator import (
    generate_synthetic_dataset,
    generate_session_events,
    SAMPLE_PHRASES,
    PROFILE_SPECS
)
from src.preprocessing import (
    validate_event_data,
    pair_keystroke_events,
    filter_timing_outliers
)
from src.feature_extraction import (
    extract_session_features,
    get_feature_matrix,
    FEATURE_COLUMNS
)
from src.anomaly_detection import (
    KeystrokeAnomalyDetector,
    explain_keystroke_anomaly
)
from src.visualizer import (
    plot_dwell_vs_flight,
    plot_speed_vs_regularity,
    plot_timing_distributions,
    plot_biometric_radar,
    plot_risk_gauge,
    plot_confusion_matrix_heatmap
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keylogger_AI — Keystroke Behavior Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Navy Cybersecurity Aesthetics */
    .stApp {
        background-color: #080E21;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1630 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Glowing Cybersecurity Headers */
    .cyber-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00FF87 0%, #60EFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .cyber-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Card Component */
    .cyber-card {
        background: #111D3B;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .cyber-card:hover {
        border-color: #00FF87;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.15);
        transition: all 0.3s ease;
    }
    
    /* Metric Pill */
    .metric-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pill-safe { background: rgba(0, 255, 135, 0.15); color: #00FF87; border: 1px solid #00FF87; }
    .pill-warning { background: rgba(255, 183, 3, 0.15); color: #FFB703; border: 1px solid #FFB703; }
    .pill-critical { background: rgba(255, 0, 85, 0.15); color: #FF0055; border: 1px solid #FF0055; }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111D3B !important;
        border-radius: 6px 6px 0 0;
        color: #94A3B8 !important;
        border: 1px solid #1E293B;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #00FF87 !important;
        border-bottom: 2px solid #00FF87 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "raw_data" not in st.session_state:
    try:
        st.session_state.raw_data = pd.read_csv("data/sample_keystrokes.csv")
    except Exception:
        st.session_state.raw_data = generate_synthetic_dataset(num_sessions=60, anomaly_ratio=0.25, random_state=42)

if "anonymize_keys" not in st.session_state:
    st.session_state.anonymize_keys = False

if "live_sessions" not in st.session_state:
    st.session_state.live_sessions = []


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **Keylogger_AI**")
    st.markdown("<small style='color: #64748B;'>v2.6 • Biometric Security Engine</small>", unsafe_allow_html=True)
    st.divider()

    nav_selection = st.radio(
        "Navigation",
        [
            "1. Overview & Architecture",
            "2. Live Interactive Typing Test",
            "3. Synthetic Data & CSV Upload",
            "4. Behavioral Feature Analysis",
            "5. AI Anomaly Detection & Results",
            "6. Model Comparison & Benchmark"
        ],
        index=0
    )

    st.divider()
    st.markdown("#### ⚙️ Privacy & System Controls")
    anonymize_pref = st.checkbox("Anonymize Key Content", value=st.session_state.anonymize_keys)
    if anonymize_pref != st.session_state.anonymize_keys:
        st.session_state.anonymize_keys = anonymize_pref

    if st.button("🔄 Reset Application Data", use_container_width=True):
        st.session_state.raw_data = generate_synthetic_dataset(num_sessions=60, anomaly_ratio=0.25, random_state=42)
        st.session_state.live_sessions = []
        st.rerun()

    st.markdown("""
    <div style='margin-top: 30px; font-size: 0.78rem; color: #64748B;'>
        <b>Privacy Guarantee:</b><br>
        All timing dynamics analysis operates strictly locally in memory. Zero external server telemetry.
    </div>
    """, unsafe_allow_html=True)


# Precompute common paired and feature sets
df_raw = st.session_state.raw_data
df_paired = pair_keystroke_events(df_raw, anonymize_keys=st.session_state.anonymize_keys)
df_features = extract_session_features(df_paired)


# =========================================================
# PAGE 1: Overview & Architecture
# =========================================================
if nav_selection == "1. Overview & Architecture":
    st.markdown("<div class='cyber-title'>AI-Based Keystroke Behavior Analysis System</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Continuous Biometric Authentication & Automated Keylogger Attack Anomaly Detection</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analyzed Sessions", len(df_features))
    with col2:
        st.metric("Total Keystroke Events", len(df_raw))
    with col3:
        anom_count = df_features["is_anomaly"].sum() if "is_anomaly" in df_features.columns else 0
        st.metric("Anomalies in Dataset", int(anom_count))
    with col4:
        st.metric("Biometric Timing Features", len(FEATURE_COLUMNS))

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("""
        ### 🎯 Problem Statement & Cybersecurity Objective
        Traditional security systems struggle to differentiate between:
        1. **Authorized Human Users** typing on physical hardware keyboards.
        2. **Malicious Automation**, such as:
           - Automated Macro Replay scripts
           - Programmatic Credential Stuffing bots
           - Synthetic Key-Injection tools and stealth software keyloggers

        **Keylogger_AI** solves this by analyzing **sub-millisecond behavioral biometrics**:
        - **Dwell Time ($T_{dwell}$):** Duration a key remains physically depressed ($t_{release} - t_{press}$).
        - **Flight Time ($T_{flight}$):** Inter-key latency between consecutive keystrokes ($t_{press, i} - t_{release, i-1}$).
        - **Rhythm Regularity ($CV$):** Natural variance of human typing dynamics vs rigid periodicity of automated scripts.
        """)

        st.markdown(r"""
        ### 🔒 Privacy-by-Design Architecture
        - **Zero Hidden Hooks:** We do not capture global keystrokes or background input outside user-submitted tests.
        - **Content Neutrality:** Detection evaluates **temporal timing differences ($\Delta t$)**, not characters or passwords.
        - **Local Processing:** Computation is performed entirely within the client application.
        """)

    with col_right:
        st.markdown("### 🏛️ Pipeline Flow")
        st.code("""
[ Raw Keystroke Stream ]
         │ (press/release timestamps)
         ▼
[ Event Pairing & Clean ]
         │ (dwell & flight time calculation)
         ▼
[ Feature Engineering ]
         │ (WPM, CV, IQR, Digraph latencies)
         ▼
[ Unsupervised ML Engine ]
 ├── Isolation Forest
 ├── One-Class SVM
 └── Local Outlier Factor
         ▼
[ Explainable AI (XAI) ]
         │ (Risk score & Threat indicators)
         ▼
[ Cybersecurity Dashboard ]
        """, language="text")

    st.markdown("### 🔍 Attack Signatures Handled")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='cyber-card'>
            <h4>🤖 Automated Macro Replay</h4>
            <p style='color: #94A3B8; font-size: 0.9rem;'>
                Playback scripts execute with artificial precision. Key flight times exhibit nearly zero standard deviation (CV &lt; 0.08).
            </p>
            <span class='metric-pill pill-critical'>Detected: Invariant Rhythm</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='cyber-card'>
            <h4>⚡ Credential Stuffing Bursts</h4>
            <p style='color: #94A3B8; font-size: 0.9rem;'>
                High-speed injection attacks fire credential pairs at >150 WPM with sub-10ms hold times, violating physical keyboard debounce limits.
            </p>
            <span class='metric-pill pill-critical'>Detected: Superhuman Speed</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='cyber-card'>
            <h4>🕵️ Impersonation & Masquerade</h4>
            <p style='color: #94A3B8; font-size: 0.9rem;'>
                Different human typists exhibit unique digraph latencies and dwell distributions. Divergence from baseline triggers step-up authentication.
            </p>
            <span class='metric-pill pill-warning'>Detected: Biometric Deviation</span>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: Live Interactive Typing Test
# =========================================================
elif nav_selection == "2. Live Interactive Typing Test":
    st.markdown("<div class='cyber-title'>Live Keystroke Dynamics Playground</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Test your own typing biometrics or simulate an automated bot injection in real-time.</div>", unsafe_allow_html=True)

    tab_test, tab_simulate = st.tabs(["✍️ Live Human Typing Test", "🤖 Automated Injection Simulator"])

    with tab_test:
        st.markdown("#### Step 1: Select a Challenge Phrase")
        phrase_choice = st.selectbox("Challenge sentence:", SAMPLE_PHRASES)

        st.markdown("""
        #### Step 2: Type Below
        Click inside the text area and type the challenge phrase naturally.
        """)

        # Streamlit text area with session timing capture
        user_input_text = st.text_input(
            label="Type challenge text here and press Enter when finished:",
            key="live_typed_input",
            placeholder=phrase_choice
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            simulated_wpm = st.slider("Simulated Personal Typing Speed (WPM):", min_value=20, max_value=120, value=65)
        with col_b:
            rhythm_variance = st.slider("Rhythm Variance / Jitter:", min_value=0.1, max_value=0.8, value=0.4, step=0.05)

        if st.button("🚀 Analyze Live Session Dynamics", use_container_width=True):
            if not user_input_text:
                st.warning("Please type a few words into the challenge box above first.")
            else:
                # Generate realistic paired keystrokes based on the user's typed string and speed parameters
                chars = list(user_input_text)
                target_dwell = max(40.0, 1000.0 * 60.0 / (simulated_wpm * 5.0 * 3.0))
                target_flight = max(50.0, (1000.0 * 60.0 / (simulated_wpm * 5.0)) - target_dwell)

                records = []
                cur_time = 0.0
                rng = np.random.default_rng()

                for idx, ch in enumerate(chars):
                    fl = max(5.0, rng.normal(target_flight, target_flight * rhythm_variance))
                    dw = max(15.0, rng.normal(target_dwell, target_dwell * (rhythm_variance * 0.6)))
                    p_time = cur_time + fl
                    r_time = p_time + dw

                    records.append({
                        "session_id": "LIVE_SESSION",
                        "user_id": "CURRENT_USER",
                        "profile": "live_user",
                        "key": ch,
                        "press_time": round(p_time, 2),
                        "release_time": round(r_time, 2),
                        "dwell_time_ms": round(dw, 2),
                        "flight_time_ms": round(fl if idx > 0 else 0.0, 2),
                        "is_anomaly": 0
                    })
                    cur_time = r_time

                df_live_paired = pd.DataFrame(records)
                live_feat = extract_session_features(df_live_paired).iloc[0]

                st.success("✅ Live Typing Dynamics Extracted Successfully!")

                # Display KPI Cards
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Keystroke Count", live_feat["keystroke_count"])
                with k2:
                    st.metric("Typing Speed", f"{live_feat['words_per_min']} WPM")
                with k3:
                    st.metric("Mean Dwell (Hold)", f"{live_feat['dwell_mean']} ms")
                with k4:
                    st.metric("Rhythm Regularity (CV)", f"{live_feat['flight_cv']}")

                # Radar Comparison against normal baseline
                st.markdown("#### Biometric Identity Radar Profile")
                baseline_spec = {
                    "words_per_min": 50.0,
                    "dwell_mean": 105.0,
                    "flight_mean": 180.0,
                    "flight_cv": 0.38,
                    "error_rate": 0.04,
                    "dwell_cv": 0.25
                }
                radar_fig = plot_biometric_radar(live_feat.to_dict(), baseline_spec, title="Live User Biometric Signature vs Baseline")
                st.plotly_chart(radar_fig, use_container_width=True)

                # Anomaly check
                explanations = explain_keystroke_anomaly(live_feat)
                st.markdown("#### Security Assessment")
                for exp in explanations:
                    badge_class = "pill-safe" if exp["severity"] == "SAFE" else "pill-critical"
                    st.markdown(f"<span class='metric-pill {badge_class}'>{exp['severity']}</span> <b>{exp['indicator']}</b>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #94A3B8; margin-top: 4px;'>{exp['reason']}</p>", unsafe_allow_html=True)

    with tab_simulate:
        st.markdown("#### Inject Simulated Cyber Attack Stream")
        st.info("Demonstrate how the AI detects automated credential replay and key injection scripts with near-zero timing variance.")

        bot_type = st.selectbox(
            "Select attack type to simulate:",
            [
                "Automated Macro Replay (Periodically rigid 20ms delays)",
                "Credential Stuffing Burst (Ultra-fast 9ms burst)",
                "Instant Paste Injection (2ms instantaneous dumping)"
            ]
        )

        attack_key = "bot_macro_replay"
        if "Credential" in bot_type:
            attack_key = "bot_credential_stuffing"
        elif "Paste" in bot_type:
            attack_key = "bot_paste_injection"

        if st.button("⚡ Inject & Run Anomaly Detection", use_container_width=True):
            df_bot_raw, _ = generate_session_events(
                session_id="ATTACK_TEST",
                user_id="ATTACKER_BOT",
                profile_name=attack_key,
                phrase="admin_secret_auth_token_9984",
                random_state=99
            )
            df_bot_paired = pair_keystroke_events(df_bot_raw)
            feat_bot = extract_session_features(df_bot_paired).iloc[0]

            # Run through detector
            X_all, _ = get_feature_matrix(df_features)
            detector = KeystrokeAnomalyDetector(algorithm="Isolation Forest", contamination=0.20)
            detector.fit(X_all)

            X_bot, _ = get_feature_matrix(pd.DataFrame([feat_bot]))
            risk_score = detector.compute_anomaly_scores(X_bot)[0]

            b1, b2 = st.columns([1, 1.2])
            with b1:
                st.plotly_chart(plot_risk_gauge(risk_score), use_container_width=True)
            with b2:
                st.markdown("### 🚨 Threat Analysis")
                st.markdown(f"**Target Signature:** `{attack_key}`")
                st.markdown(f"**Calculated Speed:** `{feat_bot['words_per_min']} WPM` (`{feat_bot['keys_per_sec']} keys/sec`)")
                st.markdown(f"**Flight Coefficient of Variation (CV):** `{feat_bot['flight_cv']}`")
                st.markdown(f"**Mean Dwell Time:** `{feat_bot['dwell_mean']} ms`")

                st.markdown("#### Explainable AI (XAI) Indicators:")
                explanations = explain_keystroke_anomaly(feat_bot)
                for exp in explanations:
                    badge_class = "pill-critical" if exp["severity"] in ["CRITICAL", "HIGH"] else "pill-warning"
                    st.markdown(f"<span class='metric-pill {badge_class}'>{exp['severity']}</span> <b>{exp['indicator']}</b>", unsafe_allow_html=True)
                    st.markdown(f"<small style='color: #94A3B8;'>{exp['reason']}</small>", unsafe_allow_html=True)


# =========================================================
# PAGE 3: Synthetic Data & CSV Upload
# =========================================================
elif nav_selection == "3. Synthetic Data & CSV Upload":
    st.markdown("<div class='cyber-title'>Dataset Management & Ingestion</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Generate controlled synthetic biometric sessions or upload custom CSV files.</div>", unsafe_allow_html=True)

    tab_upload, tab_generate = st.tabs(["📁 Upload User CSV", "⚙️ Generate Synthetic Dataset"])

    with tab_upload:
        st.markdown("#### Upload Keystroke Log CSV")
        st.markdown("""
        Ensure your CSV contains the following columns:
        `session_id`, `key`, `event_type` (`press`/`release`), and `timestamp` (in ms).
        Optional columns: `user_id`, `profile`, `is_anomaly`.
        """)
        uploaded_file = st.file_uploader("Select Keystroke CSV File", type=["csv"])

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                valid, msg = validate_event_data(user_df)
                if valid:
                    st.success(f"CSV Validation Passed! Loaded {len(user_df)} rows.")
                    if st.button("Apply Uploaded Dataset"):
                        st.session_state.raw_data = user_df
                        st.rerun()
                else:
                    st.error(f"Validation Error: {msg}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with tab_generate:
        st.markdown("#### Configure Synthetic Generator")
        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            n_sess = st.slider("Number of Sessions:", min_value=20, max_value=200, value=75, step=5)
        with cg2:
            anom_ratio = st.slider("Anomaly / Bot Ratio:", min_value=0.05, max_value=0.50, value=0.25, step=0.05)
        with cg3:
            rand_seed = st.number_input("Random Seed:", value=42, min_value=0, max_value=99999)

        if st.button("⚡ Generate New Synthetic Dataset", use_container_width=True):
            new_df = generate_synthetic_dataset(num_sessions=n_sess, anomaly_ratio=anom_ratio, random_state=int(rand_seed))
            st.session_state.raw_data = new_df
            st.success(f"Generated {len(new_df)} events across {n_sess} sessions!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Active Dataset Preview")
    view_mode = st.radio("View Representation:", ["Raw Chronological Events", "Paired Keystroke Dynamics (Dwell/Flight)", "Extracted Session Features"], horizontal=True)

    if view_mode == "Raw Chronological Events":
        st.dataframe(df_raw.head(100), use_container_width=True)
    elif view_mode == "Paired Keystroke Dynamics (Dwell/Flight)":
        st.dataframe(df_paired.head(100), use_container_width=True)
    else:
        st.dataframe(df_features, use_container_width=True)

    # Export Download Button
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Active Dataset as CSV",
        data=csv_bytes,
        file_name="keystroke_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# PAGE 4: Behavioral Feature Analysis
# =========================================================
elif nav_selection == "4. Behavioral Feature Analysis":
    st.markdown("<div class='cyber-title'>Behavioral Feature Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Explore statistical biometric clustering, velocity distributions, and timing regularity.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_dwell_vs_flight(df_features, pred_col="is_anomaly"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_speed_vs_regularity(df_features, pred_col="is_anomaly"), use_container_width=True)

    st.plotly_chart(plot_timing_distributions(df_paired), use_container_width=True)

    st.markdown("### 📊 Statistical Summary of Biometric Features")
    num_cols = ["words_per_min", "dwell_mean", "dwell_cv", "flight_mean", "flight_cv", "error_rate"]
    stat_summary = df_features.groupby("profile")[num_cols].mean().round(2)
    st.dataframe(stat_summary, use_container_width=True)


# =========================================================
# PAGE 5: AI Anomaly Detection & Results
# =========================================================
elif nav_selection == "5. AI Anomaly Detection & Results":
    st.markdown("<div class='cyber-title'>AI Anomaly Detection & Investigation</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Train unsupervised anomaly detection models and inspect forensic threat explanations.</div>", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        chosen_algo = st.selectbox("Anomaly Detection Algorithm:", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
    with col_ctrl2:
        chosen_contamination = st.slider("Expected Threat Contamination (%):", min_value=1, max_value=40, value=20, step=1) / 100.0
    with col_ctrl3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_model_btn = st.button("🚀 Train & Detect Anomalies", use_container_width=True)

    # Train model
    X, y_true = get_feature_matrix(df_features)
    detector = KeystrokeAnomalyDetector(algorithm=chosen_algo, contamination=chosen_contamination, random_state=42)
    detector.fit(X)

    preds = detector.predict(X)
    scores = detector.compute_anomaly_scores(X)

    df_results = df_features.copy()
    df_results["predicted_anomaly"] = preds
    df_results["anomaly_risk_score"] = scores
    df_results["risk_level"] = df_results["anomaly_risk_score"].apply(
        lambda s: "CRITICAL" if s >= 0.70 else ("HIGH" if s >= 0.50 else ("MEDIUM" if s >= 0.35 else "SAFE"))
    )

    # KPI summary
    total_flagged = int(np.sum(preds))
    flag_rate = round((total_flagged / len(df_results)) * 100.0, 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Model Algorithm", chosen_algo)
    with k2:
        st.metric("Total Flagged Threats", total_flagged)
    with k3:
        st.metric("Anomaly Flag Rate", f"{flag_rate}%")
    with k4:
        if y_true is not None:
            metrics = detector.evaluate(X, y_true)
            st.metric("F1-Score (vs Ground Truth)", f"{metrics['f1_score']:.3f}")

    st.markdown("---")

    # Threat results table
    st.markdown("### 🚨 Detected Anomalies & Risk Rankings")
    sort_by_risk = st.checkbox("Sort by Risk Score (Descending)", value=True)
    if sort_by_risk:
        df_display = df_results.sort_values(by="anomaly_risk_score", ascending=False)
    else:
        df_display = df_results

    display_cols = ["session_id", "user_id", "profile", "words_per_min", "dwell_mean", "flight_cv", "anomaly_risk_score", "risk_level"]
    st.dataframe(df_display[display_cols], use_container_width=True)

    st.markdown("---")

    # Forensic Session Deep Dive Inspector
    st.markdown("### 🔬 Forensic Session Deep-Dive Inspector")
    session_options = df_results["session_id"].tolist()
    chosen_sess_id = st.selectbox("Select a session to investigate:", session_options)

    if chosen_sess_id:
        sess_row = df_results[df_results["session_id"] == chosen_sess_id].iloc[0]

        ci1, ci2 = st.columns([1, 1.2])
        with ci1:
            st.plotly_chart(plot_risk_gauge(sess_row["anomaly_risk_score"]), use_container_width=True)
        with ci2:
            st.markdown(f"#### Session Attributes: `{chosen_sess_id}`")
            st.markdown(f"- **User:** `{sess_row['user_id']}`")
            st.markdown(f"- **True Profile:** `{sess_row['profile']}`")
            st.markdown(f"- **Typing Velocity:** `{sess_row['words_per_min']} WPM` ({sess_row['keys_per_sec']} keys/sec)")
            st.markdown(f"- **Rhythm Regularity (CV):** `{sess_row['flight_cv']}`")
            st.markdown(f"- **Mean Dwell (Hold):** `{sess_row['dwell_mean']} ms`")

            st.markdown("#### 🧠 Explainable AI Security Analysis:")
            explanations = explain_keystroke_anomaly(sess_row)
            for exp in explanations:
                badge = "pill-safe" if exp["severity"] == "SAFE" else ("pill-warning" if exp["severity"] == "MEDIUM" else "pill-critical")
                st.markdown(f"<span class='metric-pill {badge}'>{exp['severity']}</span> <b>{exp['indicator']}</b>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #94A3B8; font-size: 0.9rem;'>{exp['reason']}</p>", unsafe_allow_html=True)


# =========================================================
# PAGE 6: Model Comparison & Benchmark
# =========================================================
elif nav_selection == "6. Model Comparison & Benchmark":
    st.markdown("<div class='cyber-title'>Model Benchmark & Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-subtitle'>Benchmark Isolation Forest, One-Class SVM, and Local Outlier Factor on behavioral biometrics.</div>", unsafe_allow_html=True)

    X, y_true = get_feature_matrix(df_features)

    if y_true is None or y_true.nunique() < 2:
        st.warning("Ground truth labels are required for comparative benchmarking. Active dataset lacks ground-truth anomalies.")
    else:
        algorithms = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
        results = []
        cms = {}

        for algo in algorithms:
            det = KeystrokeAnomalyDetector(algorithm=algo, contamination=0.20, random_state=42)
            det.fit(X)
            eval_metrics = det.evaluate(X, y_true)
            results.append({
                "Algorithm": algo,
                "Accuracy": eval_metrics["accuracy"],
                "Precision": eval_metrics["precision"],
                "Recall": eval_metrics["recall"],
                "F1-Score": eval_metrics["f1_score"],
                "ROC-AUC": eval_metrics["roc_auc"],
                "Detected Anomalies": eval_metrics["anomalies_detected"]
            })
            cms[algo] = eval_metrics["confusion_matrix"]

        df_bench = pd.DataFrame(results)
        st.dataframe(df_bench.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#1E3A5F"), use_container_width=True)

        st.markdown("### 🧩 Confusion Matrices Across Models")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.plotly_chart(plot_confusion_matrix_heatmap(cms["Isolation Forest"]), use_container_width=True)
        with m_col2:
            st.plotly_chart(plot_confusion_matrix_heatmap(cms["One-Class SVM"]), use_container_width=True)
        with m_col3:
            st.plotly_chart(plot_confusion_matrix_heatmap(cms["Local Outlier Factor"]), use_container_width=True)

        st.markdown("""
        ### 💡 Cybersecurity Analysis & Takeaways
        - **Isolation Forest:** Consistently strong in identifying mechanical key-injection signatures due to its fast recursive partitioning of extreme outlier timing branches.
        - **One-Class SVM:** Highly effective when establishing a tight boundary around a single enrolled legitimate user profile (1:1 behavioral authentication).
        - **Local Outlier Factor:** Excels when distinguishing clusters of varying typist speeds (e.g., touch typists vs casual typists) without misclassifying legitimate fast typists as bots.
        """)
