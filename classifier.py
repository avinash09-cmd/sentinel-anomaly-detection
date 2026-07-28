"""
classifier.py
=============
Anomaly-TYPE classifier: given an event already flagged by detector.py
(high AE reconstruction error / Isolation Forest score), classify WHICH of
the 7 attack patterns it most resembles.

Design rationale:
- This is the ONLY place we touch the labeled attack set for anything beyond
  threshold calibration, and only on already-flagged events -- so severe
  class imbalance among "normal" events never enters this model at all.
  Still imbalanced across the 7 attack classes, so we use class_weight to
  avoid the model ignoring the rarest classes (e.g. device_spoofing).
- LightGBM (gradient-boosted trees) chosen because (a) it's fast enough to
  retrain on every analyst-feedback loop in production, (b) tree splits are
  naturally interpretable, and (c) it gives feature_importances_ and SHAP
  values for free -- doubling as the explainability backbone (explain.py).
- Trained on the SAME engineered features + detector scores as the
  Isolation Forest, so the "why" a SOC analyst sees lines up with what the
  detector already flagged, rather than introducing a second, disjoint
  feature space.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "hour_deviation_z", "geo_velocity_kmh", "geo_deviation_km",
    "resource_novelty", "device_mismatch", "duration_deviation_z",
    "bytes_deviation_z", "auth_fail_burst", "ae_recon_error", "iso_forest_score",
]

ATTACK_TYPES = [
    "credential_misuse", "lateral_movement", "brute_force",
    "impossible_travel", "device_spoofing", "low_and_slow_exfiltration",
    "insider_drift",
]


def train_classifier(scored_df: pd.DataFrame, test_size=0.25, seed=42):
    attack_df = scored_df[scored_df["label"] != "normal"].copy()
    attack_df[FEATURE_COLS] = attack_df[FEATURE_COLS].fillna(0.0)

    X = attack_df[FEATURE_COLS]
    y = attack_df["label"]

    # stratify keeps every attack type represented in both splits even
    # though some classes have very few samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=15,          # small trees -- dataset is small, avoid overfitting
        learning_rate=0.05,
        class_weight="balanced",  # counter imbalance ACROSS attack types
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    return model, (X_train, X_test, y_train, y_test, y_pred), report, cm


if __name__ == "__main__":
    scored_df = pd.read_parquet("data/scored_events.parquet")
    print(f"Loaded {len(scored_df)} scored events "
          f"({(scored_df['label'] != 'normal').sum()} labeled attack events).")

    model, splits, report, cm = train_classifier(scored_df)
    X_train, X_test, y_train, y_test, y_pred = splits

    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")
    print("\n--- Classification report (attack-type prediction, held-out test set) ---")
    print(report)

    print("--- Confusion matrix (rows=true, cols=pred) ---")
    labels = sorted(y_test.unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    print("\n--- Feature importances (gain) ---")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(imp.to_string())

    import joblib
    joblib.dump(model, "data/gbm_classifier.joblib")
    print("\nSaved classifier -> data/gbm_classifier.joblib")
