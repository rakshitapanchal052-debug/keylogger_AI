"""
Machine Learning Anomaly Detection Module for Keystroke Dynamics.
Implements Isolation Forest, One-Class SVM, and Local Outlier Factor algorithms,
along with risk scoring, model evaluation, and an Explainable AI (XAI) security rule engine.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
from src.feature_extraction import FEATURE_COLUMNS


class KeystrokeAnomalyDetector:
    """
    Unified detector wrapper for multiple ML anomaly detection algorithms.
    """

    def __init__(
        self,
        algorithm: str = "Isolation Forest",
        contamination: float = 0.15,
        random_state: int = 42
    ):
        self.algorithm = algorithm
        self.contamination = max(0.01, min(0.49, contamination))
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: List[str] = []
        self._init_model()

    def _init_model(self):
        if self.algorithm == "Isolation Forest":
            self.model = IsolationForest(
                n_estimators=150,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.algorithm == "One-Class SVM":
            self.model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.contamination
            )
        elif self.algorithm == "Local Outlier Factor":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def fit(self, X: pd.DataFrame) -> "KeystrokeAnomalyDetector":
        """
        Fit the scaler and anomaly detector model on feature matrix X.
        """
        self.feature_names = [col for col in FEATURE_COLUMNS if col in X.columns]
        if not self.feature_names:
            self.feature_names = list(X.columns)

        X_subset = X[self.feature_names]
        X_scaled = self.scaler.fit_transform(X_subset)
        self.model.fit(X_scaled)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary anomaly flags (1 for anomaly, 0 for normal).
        Scikit-learn returns -1 for outlier, 1 for inlier.
        """
        X_subset = X[self.feature_names]
        X_scaled = self.scaler.transform(X_subset)
        raw_preds = self.model.predict(X_scaled)
        # Convert -1 (outlier) -> 1 (anomaly), 1 (inlier) -> 0 (normal)
        return np.where(raw_preds == -1, 1, 0)

    def compute_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute continuous normalized anomaly risk scores in range [0.0, 1.0].
        Higher score = higher suspicion of anomaly / bot / macro.
        """
        X_subset = X[self.feature_names]
        X_scaled = self.scaler.transform(X_subset)

        if hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(X_scaled)
            # Lower decision_function means more abnormal.
            # Normalize with sigmoid: score = 1 / (1 + exp(raw_scores * 2))
            scores = 1.0 / (1.0 + np.exp(np.clip(raw_scores * 2.5, -15, 15)))
        elif hasattr(self.model, "score_samples"):
            raw_scores = self.model.score_samples(X_scaled)
            scores = 1.0 / (1.0 + np.exp(np.clip(raw_scores * 2.5, -15, 15)))
        else:
            scores = self.predict(X).astype(float)

        return np.round(scores, 4)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Calculate classification performance metrics against ground-truth labels.
        """
        y_pred = self.predict(X)
        scores = self.compute_anomaly_scores(X)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = 0.5

        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "total_samples": len(y_true),
            "anomalies_detected": int(np.sum(y_pred))
        }


def explain_keystroke_anomaly(row: pd.Series) -> List[Dict[str, str]]:
    """
    Rule-based Explainable AI (XAI) security engine that analyzes
    individual keystroke features and generates human-readable risk indicators.
    """
    explanations: List[Dict[str, str]] = []

    flight_cv = row.get("flight_cv", 0.5)
    dwell_cv = row.get("dwell_cv", 0.5)
    wpm = row.get("words_per_min", 40.0)
    kps = row.get("keys_per_sec", 4.0)
    dwell_mean = row.get("dwell_mean", 100.0)
    flight_mean = row.get("flight_mean", 150.0)
    error_rate = row.get("error_rate", 0.05)

    # Indicator 1: Invariant Timing (Macro / Script signature)
    if flight_cv < 0.08:
        explanations.append({
            "severity": "CRITICAL",
            "indicator": "Mechanical Periodic Cadence",
            "reason": f"Flight coefficient of variation is extremely low (CV={flight_cv:.3f} < 0.08). Legitimate human typing always exhibits natural rhythm variance. Strongly indicates automated macro replay or key injection script."
        })

    # Indicator 2: Superhuman Typing Speed
    if wpm > 140 or kps > 13.0:
        explanations.append({
            "severity": "CRITICAL",
            "indicator": "Superhuman Velocity",
            "reason": f"Typing speed is {wpm:.1f} WPM ({kps:.1f} keys/sec), surpassing professional human touch-typist limits. Typical of programmatic paste or keylogger test replay attacks."
        })

    # Indicator 3: Ultra-low Dwell Time
    if dwell_mean < 25.0:
        explanations.append({
            "severity": "HIGH",
            "indicator": "Sub-threshold Key Hold Duration",
            "reason": f"Average key dwell time is {dwell_mean:.1f} ms. Physical membrane and mechanical keyboard debounce delays usually keep human dwell times above 40ms. Likely synthetic virtual key generation."
        })

    # Indicator 4: Inhuman Flight Time
    if flight_mean < 20.0:
        explanations.append({
            "severity": "HIGH",
            "indicator": "Near-Zero Inter-Key Latency",
            "reason": f"Average flight time between keystrokes is {flight_mean:.1f} ms. Incompatible with human biomechanical finger movement transitions across key caps."
        })

    # Indicator 5: Unnatural Consistency
    if dwell_cv < 0.06:
        explanations.append({
            "severity": "MEDIUM",
            "indicator": "Rigid Hold Consistency",
            "reason": f"Key hold variance is virtually absent (dwell CV={dwell_cv:.3f}). Suggests uniform synthetic timer."
        })

    # Indicator 6: Zero Error Rate at High Speed
    if error_rate == 0.0 and wpm > 85:
        explanations.append({
            "severity": "LOW",
            "indicator": "Flawless High-Velocity Execution",
            "reason": "Zero backspaces recorded across a high-speed session. While possible for expert typists, reinforces suspicion when paired with low timing variance."
        })

    if not explanations:
        explanations.append({
            "severity": "SAFE",
            "indicator": "Legitimate Human Biometric Signature",
            "reason": f"Natural rhythm variance (Flight CV={flight_cv:.2f}), normal human velocity ({wpm:.1f} WPM), and realistic dwell distribution ({dwell_mean:.1f} ms) consistent with genuine human behavior."
        })

    return explanations
