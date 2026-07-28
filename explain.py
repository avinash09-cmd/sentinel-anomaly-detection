"""
explain.py
==========
Turns a flagged event + its GBM attack-type prediction into a structured,
human-readable alert object for the SOC dashboard.

Design rationale:
- Tree-based SHAP is used because it's exact and fast for GBMs (no sampling
  approximation needed, unlike KernelSHAP) -- important for a live dashboard
  where alerts must explain themselves in near real-time.
- We map engineered feature names -> analyst-friendly phrases once, here, so
  every downstream consumer (dashboard, evaluate.py, alert JSON) gets
  consistent language.
- Output schema matches the brief exactly: risk_score, predicted_type,
  top_contributing_features, entity_history_snippet.
"""

import numpy as np
import pandas as pd
import shap

FEATURE_COLS = [
    "hour_deviation_z", "geo_velocity_kmh", "geo_deviation_km",
    "resource_novelty", "device_mismatch", "duration_deviation_z",
    "bytes_deviation_z", "auth_fail_burst", "ae_recon_error", "iso_forest_score",
]

# Human-readable phrasing per feature, used to render SHAP's top contributors
# as a sentence instead of a raw feature name + number.
FEATURE_PHRASES = {
    "hour_deviation_z": "unusual login hour for this entity",
    "geo_velocity_kmh": "geo-velocity spike (impossible travel)",
    "geo_deviation_km": "access far outside this entity's normal geographic radius",
    "resource_novelty": "access to a resource this entity rarely/never uses",
    "device_mismatch": "first-seen or unrecognized device fingerprint",
    "duration_deviation_z": "session duration far from this entity's norm",
    "bytes_deviation_z": "data volume transferred far above this entity's norm",
    "auth_fail_burst": "burst of authentication failures in a short window",
    "ae_recon_error": "sequence-model reconstruction error (overall behavioral deviation)",
    "iso_forest_score": "isolation-forest outlier score (feature-space deviation)",
}


class AlertExplainer:
    def __init__(self, gbm_model, feature_cols=FEATURE_COLS):
        self.model = gbm_model
        self.feature_cols = feature_cols
        self.explainer = shap.TreeExplainer(gbm_model)

    def explain_batch(self, X: pd.DataFrame):
        """Returns (shap_values, predicted_class_idx, predicted_proba) for a
        batch of already-flagged events."""
        shap_values = self.explainer.shap_values(X)   # list[n_classes] of (N, F) OR (N, F, C)
        proba = self.model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return shap_values, pred_idx, proba

    def _shap_for_predicted_class(self, shap_values, row_idx, class_idx):
        # LightGBM sklearn multi-class SHAP can come back as (N, F, C) or a
        # list of C arrays each (N, F) depending on version -- handle both.
        if isinstance(shap_values, list):
            return shap_values[class_idx][row_idx]
        else:
            return shap_values[row_idx, :, class_idx]

    def build_alert(self, event_row: pd.Series, X_row: pd.DataFrame, shap_values,
                     row_idx, pred_idx, proba, class_labels, entity_history: pd.DataFrame,
                     risk_score: float, top_k=3):
        class_idx = pred_idx[row_idx]
        predicted_type = class_labels[class_idx]
        confidence = float(proba[row_idx, class_idx])

        contrib = self._shap_for_predicted_class(shap_values, row_idx, class_idx)
        contrib_series = pd.Series(contrib, index=self.feature_cols)
        top_features = contrib_series.abs().sort_values(ascending=False).head(top_k)

        top_contributing_features = []
        for feat_name, _ in top_features.items():
            direction = "increased" if contrib_series[feat_name] > 0 else "decreased"
            top_contributing_features.append({
                "feature": feat_name,
                "reason": FEATURE_PHRASES.get(feat_name, feat_name),
                "shap_contribution": round(float(contrib_series[feat_name]), 4),
                "raw_value": round(float(X_row[feat_name].values[0]), 3),
                "direction": direction,
            })

        history_snippet = entity_history.tail(5)[
            ["timestamp", "resource_accessed", "geo_city", "device_fingerprint", "auth_result"]
        ].to_dict(orient="records")

        explanation_sentence = self._render_sentence(predicted_type, top_contributing_features)

        return {
            "event_id": event_row["event_id"],
            "entity_id": event_row["entity_id"],
            "entity_type": event_row["entity_type"],
            "timestamp": str(event_row["timestamp"]),
            "risk_score": round(float(risk_score), 4),
            "predicted_type": predicted_type,
            "confidence": round(confidence, 3),
            "top_contributing_features": top_contributing_features,
            "entity_history_snippet": history_snippet,
            "explanation": explanation_sentence,
        }

    @staticmethod
    def _render_sentence(predicted_type, top_features):
        reasons = " + ".join(f["reason"] for f in top_features)
        pretty_type = predicted_type.replace("_", " ")
        return f"Flagged as likely {pretty_type} due to {reasons}."


def explain_alerts(scored_df: pd.DataFrame, gbm_model, class_labels, top_n_alerts=None,
                    risk_col="ae_recon_error"):
    """Convenience driver: takes the full scored dataframe, restricts to the
    flagged subset (or top_n_alerts by risk_col if given), and returns a list
    of structured alert dicts ready for the dashboard / JSON export."""
    df = scored_df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    if top_n_alerts is not None:
        flagged = df.sort_values(risk_col, ascending=False).head(top_n_alerts)
    else:
        flagged = df

    explainer = AlertExplainer(gbm_model)
    X = flagged[FEATURE_COLS]
    shap_values, pred_idx, proba = explainer.explain_batch(X)

    alerts = []
    flagged = flagged.reset_index(drop=True)
    for i in range(len(flagged)):
        event_row = flagged.iloc[i]
        X_row = X.iloc[[i]]
        entity_hist = df[(df["entity_id"] == event_row["entity_id"]) &
                          (df["timestamp"] <= event_row["timestamp"])]
        alert = explainer.build_alert(
            event_row, X_row, shap_values, i, pred_idx, proba, class_labels,
            entity_hist, risk_score=event_row[risk_col],
        )
        alerts.append(alert)
    return alerts


if __name__ == "__main__":
    import json
    import joblib

    scored_df = pd.read_parquet("data/scored_events.parquet")
    model = joblib.load("data/gbm_classifier.joblib")

    # class_labels must match the order the sklearn classifier learned
    class_labels = list(model.classes_)

    print("Generating explained alerts for the top 15 highest-risk events...")
    alerts = explain_alerts(scored_df, model, class_labels, top_n_alerts=15)

    for a in alerts[:5]:
        print("\n" + "=" * 70)
        print(f"Entity: {a['entity_id']}  |  Risk: {a['risk_score']}  |  "
              f"Predicted: {a['predicted_type']} ({a['confidence']*100:.1f}% conf)")
        print(f"Explanation: {a['explanation']}")
        print("Top contributing features:")
        for f in a["top_contributing_features"]:
            print(f"  - {f['reason']} (shap={f['shap_contribution']}, value={f['raw_value']})")

    with open("data/sample_alerts.json", "w") as f:
        json.dump(alerts, f, indent=2, default=str)
    print(f"\nSaved {len(alerts)} structured alerts -> data/sample_alerts.json")
