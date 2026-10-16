# Keylogger_AI — AI-Based Keystroke Behavior Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-00FF87.svg?style=flat)](LICENSE)

> **Keylogger_AI** is a modern, privacy-preserving cybersecurity demonstration platform that uses **unsupervised machine learning** and **keystroke behavioral biometrics** to detect automated keylogger attacks, programmatic macro replays, and credential stuffing injections.

---

## 📑 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Comparison with Reference Project](#2-comparison-with-reference-project)
- [3. Key Features](#3-key-features)
- [4. Biometric Theory & Threat Signatures](#4-biometric-theory--threat-signatures)
- [5. System Architecture](#5-system-architecture)
- [6. Project Structure](#6-project-structure)
- [7. Installation & Setup](#7-installation--setup)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [8. Running the Application](#8-running-the-application)
- [9. Testing](#9-testing)
- [10. Safety, Privacy & Ethics](#10-safety-privacy--ethics)
- [11. Proposed Enhancements & Roadmap](#11-proposed-enhancements--roadmap)
- [12. GitHub Repository Setup Guide](#12-github-repository-setup-guide)
- [13. License](#13-license)

---

## 1. Project Overview

Modern zero-trust security and continuous authentication models increasingly rely on **behavioral biometrics** rather than static passwords alone. Even when an adversary obtains legitimate credentials, their typing patterns and automated injection tools leave distinct temporal signatures.

**Keylogger_AI** demonstrates how machine learning models (Isolation Forest, One-Class SVM, and Local Outlier Factor) analyze **sub-millisecond keystroke temporal dynamics** to detect malicious automated tools, macro scripts, and impersonation attempts without capturing sensitive personal data.

---

## 2. Comparison with Reference Project

| Dimension | Reference Repository (`10canto/Keylogger_AI`) | This Project (`Keylogger_AI Keystroke Behavior Analysis`) |
| :--- | :--- | :--- |
| **Primary Scope** | Windows desktop process scanner checking for hook APIs | Behavioral keystroke dynamics & AI anomaly detection |
| **Interface** | Legacy Tkinter desktop GUI | Responsive, modern dark-navy Streamlit dashboard |
| **Operating System** | Windows-only (`pywin32`, `psutil`, native Windows APIs) | Cross-platform (Windows, Linux, macOS, Cloud) |
| **ML Models** | Offline pre-trained classifier | Real-time Isolation Forest, One-Class SVM, and LOF |
| **Interactive Test** | None (background process monitoring) | Live typing test playground + Bot attack simulator |
| **Explainability** | Static rule tags | Dynamic Explainable AI (XAI) security indicator engine |
| **Privacy Mode** | Inspects OS-wide system process metadata | Local analysis; content-blind timing metrics ($\Delta t$) |

---

## 3. Key Features

- 🛡️ **Cybersecurity Dashboard:** Dark navy aesthetic (`#0B132B`) with cyber green accents (`#00FF87`), metric KPI cards, and responsive layout.
- ✍️ **Live Interactive Typing Playground:** Voluntary browser typing test capturing real-time cadence, dwell times, and flight intervals.
- 🤖 **Automated Attack Simulator:** Injects realistic macro replays, credential stuffing bursts, and paste attacks to evaluate model response.
- ⚙️ **Synthetic Dataset Generator:** Configurable multi-user generator simulating casual typists, touch typists, hesitant typists, and robotic scripts.
- 📁 **CSV Upload & Ingestion:** Upload custom non-sensitive event logs with schema validation.
- 🔬 **Biometric Feature Extraction:** Computes dwell time ($T_{hold}$), flight time ($T_{flight}$), velocity (WPM / KPS), coefficient of variation ($CV$), and digraph latencies.
- 🧠 **Multi-Model Anomaly Detection:**
  - **Isolation Forest:** Tree partitioning for outlier separation.
  - **One-Class SVM:** Non-linear decision boundaries for user profile modeling.
  - **Local Outlier Factor (LOF):** Density-based local anomaly ranking.
- 📊 **Explainable AI (XAI) Engine:** Translates raw statistical deviations into actionable security findings.
- 🧩 **Comparative Benchmarking:** Side-by-side evaluation with Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.

---

## 4. Biometric Theory & Threat Signatures

### Keystroke Biometric Metrics

$$\text{Dwell Time (Hold Time)} = t_{\text{release}} - t_{\text{press}}$$

$$\text{Flight Time (Inter-Key Latency)} = t_{\text{press}, i} - t_{\text{release}, i-1}$$

$$\text{Rhythm Regularity (Coefficient of Variation)}: CV = \frac{\sigma}{\mu}$$

### Threat Signatures Detected

1. **Mechanical Periodic Cadence ($CV_{\text{flight}} < 0.08$):** Human typists always demonstrate natural variance. An automated script or macro replay exhibits rigid timing.
2. **Superhuman Velocity ($> 140$ WPM / $> 13$ KPS):** Inhuman keystroke submission characteristic of automated credential stuffing.
3. **Sub-Threshold Dwell Time ($< 25$ ms):** Physical keyboard membrane debounce delays prevent human dwell times under ~30ms. Virtual programmatic keystrokes typically bypass physical debounce.
4. **Instantaneous Flight Times ($< 10$ ms):** Programmatic key-injection bypasses finger travel physics.

---

## 5. System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Data Ingestion & Event Stream              │
│  (Synthetic Generator, Live Typing Test, CSV Upload)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Preprocessing & Event Pairing              │
│      (Press/Release Matcher, Outlier Filtering)        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              Feature Extraction Engine                 │
│    (Dwell/Flight Stats, WPM, Rhythm CV, Digraphs)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Machine Learning Anomaly Engine             │
│   ├── Isolation Forest                                 │
│   ├── One-Class SVM                                    │
│   └── Local Outlier Factor                             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Explainable AI (XAI) Security Rule          │
│        (Threat Levels, Indicators, Risk Gauges)        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Modern Streamlit Dashboard UI              │
└────────────────────────────────────────────────────────┘
```

---

## 6. Project Structure

```
Keylogger_AI/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Lightweight project dependencies
├── README.md                   # Comprehensive documentation
├── .gitignore                  # Git ignore rules for Python/IDEs/secrets
│
├── data/
│   └── sample_keystrokes.csv   # Pre-generated sample keystroke dataset
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_generator.py       # Synthetic keystroke & attack generator
│   ├── preprocessing.py        # Event pairing, dwell/flight calculator
│   ├── feature_extraction.py   # Statistical biometric feature extractor
│   ├── anomaly_detection.py    # Isolation Forest, SVM, LOF & XAI engine
│   └── visualizer.py           # Cyber-themed Plotly charts
│
└── tests/
    ├── __init__.py             # Package marker
    ├── test_preprocessing.py   # Event pairing & validation tests
    └── test_anomaly_detection.py # ML model & XAI unit tests
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8 to 3.14 installed
- Git installed

### Windows

```powershell
# 1. Clone or navigate to the project directory
cd Keylogger_AI

# 2. Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Linux / macOS

```bash
# 1. Clone or navigate to the project directory
cd Keylogger_AI

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 8. Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Or run via Python module:

```bash
python -m streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 9. Testing

Run the automated test suite with `pytest`:

```bash
python -m pytest tests -v
```

Expected output:
```
tests/test_anomaly_detection.py::test_isolation_forest_fit_predict PASSED
tests/test_anomaly_detection.py::test_one_class_svm PASSED
tests/test_anomaly_detection.py::test_local_outlier_factor PASSED
tests/test_anomaly_detection.py::test_model_evaluation PASSED
tests/test_anomaly_detection.py::test_explain_keystroke_anomaly_bot PASSED
tests/test_anomaly_detection.py::test_explain_keystroke_anomaly_human PASSED
tests/test_preprocessing.py::test_validate_event_data_valid PASSED
tests/test_preprocessing.py::test_validate_event_data_missing_column PASSED
tests/test_preprocessing.py::test_pair_keystroke_events PASSED
tests/test_preprocessing.py::test_filter_timing_outliers PASSED
tests/test_preprocessing.py::test_synthetic_session_generation PASSED

============================= 11 passed in 15.17s =============================
```

---

## 10. Safety, Privacy & Ethics

- 🛡️ **No Background Surveillance:** This application does not install global system hooks, DLL injections, or background loggers.
- 🔒 **Zero Sensitive Data Ingestion:** The system only processes non-sensitive test phrases or explicitly submitted data.
- 👤 **Content Masking Option:** Users can enable the "Anonymize Key Content" toggle in the sidebar to mask key values into generic tokens while preserving timing intervals.
- 💻 **100% Local Execution:** No telemetry or timing data is transmitted over the network.

---

## 11. Proposed Enhancements & Roadmap

Future features that can expand this project:
1. **Multi-Modal Biometrics (Mouse Dynamics):** Integrating cursor curvature, jitter, and click hold time to achieve >95% verification accuracy.
2. **Deep Learning Biometric Encoders (1D-CNN / LSTM):** Sequence-based neural networks modeling long-term typing cadence.
3. **Adaptive Continuous Enrollment:** Dynamic profile updating where the model learns user typing fatigue over the course of a day.
4. **Zero-Trust PAM Integration:** Emitting step-up authentication triggers (e.g. FIDO2 / WebAuthn prompt) whenever anomaly risk exceeds 70%.

---

## 12. GitHub Repository Setup Guide

To initialize this project as a new Git repository and publish it to your GitHub account:

```bash
# 1. Initialize local Git repository
git init

# 2. Stage all files
git add .

# 3. Create initial commit
git commit -m "Initial commit: AI-Based Keystroke Behavior Analysis System"

# 4. Rename default branch to main
git branch -M main

# 5. Link your GitHub remote repository (replace with your GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/Keylogger_AI.git

# 6. Push to GitHub
git push -u origin main
```

---

## 13. License

Distributed under the MIT License. See `LICENSE` for details.
