"""
dashboard.py
============
Streamlit SOC analyst console. Dark, futuristic UI over the outputs of
data_generator.py -> profiler.py -> detector.py -> classifier.py -> explain.py.

Design rationale:
- Everything expensive (loading events, running SHAP, training-time
  artifacts) is st.cache_data / st.cache_resource'd so the "live" feel comes
  from re-rendering fast on top of precomputed scores, not from re-running
  models on every rerun -- this mirrors a real deployment where scoring
  happens in a streaming backend and the dashboard just renders alerts.
- The live-stream simulation replays already-scored events in timestamp
  order via the streaming generator, revealing them into the alert queue a
  few at a time -- giving the "SOC watching new alerts arrive" feel without
  needing a real message bus for the demo.
"""

import time
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

from explain import explain_alerts, FEATURE_COLS, FEATURE_PHRASES

st.set_page_config(
    page_title="SENTINEL // Behavioral Anomaly Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark / futuristic theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0a0e14;
    --panel: #10161f;
    --panel-border: #1e2a38;
    --accent: #00e5ff;
    --accent2: #ff2d6f;
    --text: #d6e2ea;
    --muted: #6b7f91;
}
.stApp { background-color: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--panel-border); }
h1, h2, h3 { color: var(--accent) !important; letter-spacing: 0.5px; }
div[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 10px; padding: 12px 16px;
}
div[data-testid="stMetricValue"] { color: var(--accent); font-family: 'Courier New', monospace; }
.risk-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-weight: 600; font-size: 0.78rem; font-family: monospace;
}
.risk-critical { background: rgba(255,45,111,0.18); color: #ff5c88; border: 1px solid #ff2d6f; }
.risk-high { background: rgba(255,159,10,0.18); color: #ffb347; border: 1px solid #ff9f0a; }
.risk-medium { background: rgba(0,229,255,0.14); color: #5be0f5; border: 1px solid #00e5ff; }
.alert-card {
    background: var(--panel); border: 1px solid var(--panel-border);
    border-left: 3px solid var(--accent2); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 10px;
}
.live-dot {
    height: 9px; width: 9px; background-color: #00ff88; border-radius: 50%;
    display: inline-block; margin-right: 6px; box-shadow: 0 0 8px #00ff88;
    animation: pulse 1.4s infinite;
}
@keyframes pulse { 0% {opacity:1;} 50% {opacity:0.3;} 100% {opacity:1;} }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_scored_events():
    df = pd.read_parquet("data/scored_events.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_classifier():
    return joblib.load("data/gbm_classifier.joblib")


@st.cache_data
def compute_alerts(_scored_df_hash, top_n=250):
    """Cached on a hash of the dataframe id so it only reruns SHAP once per
    session, not on every widget interaction."""
    scored_df = load_scored_events()
    model = load_classifier()
    class_labels = list(model.classes_)
    alerts = explain_alerts(scored_df, model, class_labels, top_n_alerts=top_n)
    return pd.DataFrame(alerts)


def risk_badge(score, hi=8.0, crit=15.0):
    if score >= crit:
        return f'<span class="risk-badge risk-critical">CRITICAL {score:.1f}</span>'
    elif score >= hi:
        return f'<span class="risk-badge risk-high">HIGH {score:.1f}</span>'
    else:
        return f'<span class="risk-badge risk-medium">MEDIUM {score:.1f}</span>'


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

scored_df = load_scored_events()
model = load_classifier()
alerts_df = compute_alerts(id(scored_df), top_n=250)

# ---------------------------------------------------------------------------
# Sidebar: filters + live stream control
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🛡️ SENTINEL")
st.sidebar.caption("Behavioral Anomaly Detection Console")
st.sidebar.divider()

st.sidebar.markdown("### Live Feed")
live_mode = st.sidebar.toggle("Simulate live stream", value=False)
if "stream_ptr" not in st.session_state:
    st.session_state.stream_ptr = max(20, int(len(alerts_df) * 0.15))

if live_mode:
    st.sidebar.markdown('<span class="live-dot"></span> **STREAMING**', unsafe_allow_html=True)
    speed = st.sidebar.slider("Reveal speed (alerts/tick)", 1, 10, 3)
else:
    st.sidebar.caption("Paused — showing full alert history")

st.sidebar.divider()
st.sidebar.markdown("### Filters")
entity_types = st.sidebar.multiselect(
    "Entity type", sorted(alerts_df["entity_type"].unique()),
    default=sorted(alerts_df["entity_type"].unique())
)
attack_types = st.sidebar.multiselect(
    "Predicted anomaly type", sorted(alerts_df["predicted_type"].unique()),
    default=sorted(alerts_df["predicted_type"].unique())
)
min_risk = st.sidebar.slider(
    "Minimum risk score", 0.0, float(alerts_df["risk_score"].max()), 0.0
)

# ---------------------------------------------------------------------------
# Header + KPI bar
# ---------------------------------------------------------------------------

st.markdown("# SENTINEL — Behavioral Anomaly Console")
st.caption("Explainable, real-time-style intrusion & insider-threat detection over user, "
           "service-account, and edge-device telemetry.")

visible_n = st.session_state.stream_ptr if live_mode else len(alerts_df)
visible_alerts = alerts_df.iloc[:visible_n]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Alerts (visible)", f"{len(visible_alerts):,}")
est_fp_rate = 0.023  # from evaluate.py Isolation-Forest false_positive_rate at 1% budget
k2.metric("Est. false-positive rate", f"{est_fp_rate*100:.1f}%")
k3.metric("Entities monitored", f"{scored_df['entity_id'].nunique():,}")
cold_share = scored_df["is_cold_start"].mean() * 100
k4.metric("Cold-start entities", f"{cold_share:.1f}%")
drift_indicator = "STABLE" if cold_share < 20 else "ADAPTING"
k5.metric("Model drift status", drift_indicator, delta=None)

st.divider()

# ---------------------------------------------------------------------------
# Alert queue
# ---------------------------------------------------------------------------

left, right = st.columns([1.4, 1])

with left:
    st.markdown("### 🚨 Ranked Alert Queue")
    filtered = visible_alerts[
        visible_alerts["entity_type"].isin(entity_types)
        & visible_alerts["predicted_type"].isin(attack_types)
        & (visible_alerts["risk_score"] >= min_risk)
    ].sort_values("risk_score", ascending=False)

    st.caption(f"{len(filtered)} alerts match current filters")

    if len(filtered) == 0:
        st.info("No alerts match the current filters.")
        selected_event_id = None
    else:
        display_df = filtered[["entity_id", "entity_type", "predicted_type",
                                "confidence", "risk_score", "timestamp"]].copy()
        display_df.columns = ["Entity", "Type", "Predicted Anomaly", "Confidence",
                               "Risk Score", "Timestamp"]
        display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
        display_df["Risk Score"] = display_df["Risk Score"].round(2)

        event = st.dataframe(
            display_df,
            width='stretch',
            height=520,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event.selection and event.selection.get("rows"):
            sel_row = event.selection["rows"][0]
            selected_event_id = filtered.iloc[sel_row]["event_id"]
        else:
            selected_event_id = filtered.iloc[0]["event_id"]

with right:
    st.markdown("### 🔎 Alert Drill-Down")
    if selected_event_id is None:
        st.info("Select an alert to inspect it.")
    else:
        alert = alerts_df[alerts_df["event_id"] == selected_event_id].iloc[0]

        st.markdown(
            f"""<div class="alert-card">
            <b>{alert['entity_id']}</b> ({alert['entity_type']}) &nbsp;
            {risk_badge(alert['risk_score'])}<br><br>
            <b>Predicted:</b> {alert['predicted_type'].replace('_',' ').title()}
            &nbsp;({alert['confidence']*100:.1f}% confidence)<br>
            <b>When:</b> {alert['timestamp']}<br><br>
            <i>{alert['explanation']}</i>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Top contributing factors**")
        contrib = pd.DataFrame(alert["top_contributing_features"])
        fig = go.Figure(go.Bar(
            x=contrib["shap_contribution"],
            y=[FEATURE_PHRASES.get(f, f) for f in contrib["feature"]],
            orientation="h",
            marker_color=["#ff2d6f" if v > 0 else "#00e5ff" for v in contrib["shap_contribution"]],
        ))
        fig.update_layout(
            paper_bgcolor="#10161f", plot_bgcolor="#10161f",
            font_color="#d6e2ea", height=220, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="SHAP contribution to predicted class",
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("**Entity behavioral timeline vs. normal profile**")
        ent_hist = scored_df[scored_df["entity_id"] == alert["entity_id"]].sort_values("timestamp")
        baseline_mean = ent_hist["bytes_transferred"].median()
        baseline_std = ent_hist["bytes_transferred"].std() or 1.0

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ent_hist["timestamp"], y=ent_hist["bytes_transferred"],
            mode="lines+markers", name="bytes transferred",
            line=dict(color="#00e5ff", width=1.5), marker=dict(size=4),
        ))
        fig2.add_hrect(
            y0=max(0, baseline_mean - baseline_std), y1=baseline_mean + baseline_std,
            fillcolor="#00e5ff", opacity=0.08, line_width=0,
        )
        flagged_pt = ent_hist[ent_hist["event_id"] == selected_event_id]
        if len(flagged_pt):
            fig2.add_trace(go.Scatter(
                x=flagged_pt["timestamp"], y=flagged_pt["bytes_transferred"],
                mode="markers", name="this alert",
                marker=dict(size=13, color="#ff2d6f", symbol="x"),
            ))
        fig2.update_layout(
            paper_bgcolor="#10161f", plot_bgcolor="#10161f", font_color="#d6e2ea",
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig2, width='stretch')

        with st.expander("Recent raw session history"):
            st.dataframe(
                pd.DataFrame(alert["entity_history_snippet"]),
                width='stretch', hide_index=True,
            )

st.divider()

# ---------------------------------------------------------------------------
# Distribution overview
# ---------------------------------------------------------------------------

st.markdown("### 📊 Alert Distribution")
c1, c2 = st.columns(2)
with c1:
    type_counts = visible_alerts["predicted_type"].value_counts()
    fig3 = go.Figure(go.Bar(
        x=type_counts.index.str.replace("_", " "), y=type_counts.values,
        marker_color="#00e5ff",
    ))
    fig3.update_layout(
        title="Alerts by predicted anomaly type",
        paper_bgcolor="#10161f", plot_bgcolor="#10161f", font_color="#d6e2ea",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig3, width='stretch')

with c2:
    fig4 = go.Figure(go.Histogram(
        x=visible_alerts["risk_score"], nbinsx=30, marker_color="#ff2d6f",
    ))
    fig4.update_layout(
        title="Risk score distribution",
        paper_bgcolor="#10161f", plot_bgcolor="#10161f", font_color="#d6e2ea",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig4, width='stretch')

# ---------------------------------------------------------------------------
# Live-stream auto-advance
# ---------------------------------------------------------------------------

if live_mode and st.session_state.stream_ptr < len(alerts_df):
    time.sleep(1.2)
    st.session_state.stream_ptr = min(
        len(alerts_df), st.session_state.stream_ptr + speed
    )
    st.rerun()
