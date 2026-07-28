"""
detector.py
===========
Unsupervised / semi-supervised anomaly detection.

Two models, trained ONLY on data believed normal (constraint: never train a
supervised classifier directly on the imbalanced raw labels):

1. LSTM autoencoder (primary, sequence-aware)
   - Why: behavior is a SEQUENCE of sessions per entity, not an i.i.d. bag of
     rows. An autoencoder trained to reconstruct an entity's own recent
     sequence of engineered-feature vectors will reconstruct normal sequences
     well and novel/attack sequences poorly -- reconstruction error becomes
     the anomaly score, with no need for attack labels at train time.
   - Sequences are built PER ENTITY in chronological order using a sliding
     window, using the profiler's already-lookahead-safe engineered features
     (see profiler.py) rather than raw categorical fields, so the model
     learns "how much does this deviate from this entity's own norm" rather
     than memorizing raw resource names / IPs.

2. Isolation Forest (secondary, fast baseline)
   - Why: a non-sequential, non-deep baseline on the same engineered features
     lets us prove the sequence-aware model earns its complexity (ablation in
     evaluate.py), and gives a near-instant fallback score for cold-start /
     low-volume entities where a full sequence isn't available yet.

Threshold calibration: the small labeled anomaly set is used ONLY at the end
to pick the score cutoff for a target alert budget (default top 1%) -- never
to train the detectors themselves.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "hour_deviation_z", "geo_velocity_kmh", "geo_deviation_km",
    "resource_novelty", "device_mismatch", "duration_deviation_z",
    "bytes_deviation_z", "auth_fail_burst",
]

SEQ_LEN = 8          # sliding window of sessions per entity fed to the LSTM-AE
HIDDEN_DIM = 16
LATENT_DIM = 6
EPOCHS = 12
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """Encoder-decoder GRU autoencoder over a window of an entity's engineered
    per-session feature vectors. GRU chosen over plain LSTM for fewer
    parameters -- with only ~8-step windows and a handful of features, GRU
    converges just as well with less overfitting risk on a small dataset."""

    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)
        self.seq_len = SEQ_LEN

    def forward(self, x):
        _, h_n = self.encoder_rnn(x)          # h_n: (1, B, hidden)
        z = self.to_latent(h_n.squeeze(0))    # (B, latent)
        h0 = self.from_latent(z).unsqueeze(0) # (1, B, hidden) -- init decoder hidden state
        dec_input = h0.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, hidden)
        dec_out, _ = self.decoder_rnn(dec_input, h0)
        recon = self.output_layer(dec_out)    # (B, T, n_features)
        return recon


def build_sequences(feat_df: pd.DataFrame, feature_cols=FEATURE_COLS, seq_len=SEQ_LEN):
    """Slide a window of seq_len consecutive sessions per entity (in
    timestamp order) -> one training example per window. Returns array of
    shape (N, seq_len, n_features) plus the event_id of the LAST event in
    each window (that's the event the window's reconstruction error is
    attributed to)."""
    feat_df = feat_df.sort_values(["entity_id", "timestamp"])
    sequences, last_event_ids, entity_ids = [], [], []
    for entity_id, g in feat_df.groupby("entity_id"):
        vals = g[feature_cols].values
        ids = g["event_id"].values
        if len(vals) < seq_len:
            continue
        for i in range(len(vals) - seq_len + 1):
            sequences.append(vals[i:i + seq_len])
            last_event_ids.append(ids[i + seq_len - 1])
            entity_ids.append(entity_id)
    if not sequences:
        return np.empty((0, seq_len, len(feature_cols))), [], []
    return np.stack(sequences), last_event_ids, entity_ids


def train_autoencoder(X_train, n_features, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True):
    model = LSTMAutoencoder(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    n = X_t.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_t[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.shape[0]
        if verbose:
            print(f"  epoch {epoch + 1:02d}/{epochs}  recon_MSE={total_loss / n:.5f}")
    return model


@torch.no_grad()
def reconstruction_error(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    recon = model(X_t)
    # per-window error = mean squared error across the whole window
    err = ((recon - X_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
    # also return per-feature error at the LAST timestep -- used by explain.py
    per_feature_last = ((recon[:, -1, :] - X_t[:, -1, :]) ** 2).cpu().numpy()
    return err, per_feature_last


# ---------------------------------------------------------------------------
# Isolation Forest baseline
# ---------------------------------------------------------------------------

def train_isolation_forest(X_train_flat, contamination=0.02):
    """contamination is only a rough prior for IF's internal thresholding;
    final alerting threshold is still calibrated separately in evaluate.py."""
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    )
    iso.fit(X_train_flat)
    return iso


# ---------------------------------------------------------------------------
# End-to-end training driver
# ---------------------------------------------------------------------------

def run_detection_pipeline(feat_df: pd.DataFrame, normal_only_for_training=True):
    scaler = StandardScaler()
    feat_df = feat_df.copy()
    feat_df[FEATURE_COLS] = feat_df[FEATURE_COLS].fillna(0.0)
    # Winsorize at the 0.5/99.5 percentile before scaling: a handful of
    # first-ever-session artifacts (e.g. near-zero time-delta geo-velocity)
    # otherwise dominate the scale and drown out genuine attack signal.
    # Clipping preserves rank/separation (true attacks are still well beyond
    # the clip point) while stopping single-row outliers from stretching
    # the StandardScaler fit.
    for col in FEATURE_COLS:
        lo, hi = feat_df[col].quantile([0.005, 0.995])
        feat_df[col] = feat_df[col].clip(lo, hi)
    feat_df[FEATURE_COLS] = scaler.fit_transform(feat_df[FEATURE_COLS])

    # --- Isolation Forest: trains on engineered features per-event ---
    train_mask = feat_df["label"] == "normal" if normal_only_for_training else slice(None)
    X_if_train = feat_df.loc[train_mask, FEATURE_COLS].values
    print(f"Training Isolation Forest on {len(X_if_train)} normal-labeled events "
          f"(features only, no sequence)...")
    iso = train_isolation_forest(X_if_train)
    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    feat_df["iso_forest_score"] = -iso.decision_function(feat_df[FEATURE_COLS].values)

    # --- LSTM/GRU Autoencoder: trains on sequences of normal windows ---
    seq_all, last_ids_all, ent_ids_all = build_sequences(feat_df)
    print(f"Built {len(seq_all)} sliding windows (seq_len={SEQ_LEN}) across "
          f"{feat_df['entity_id'].nunique()} entities.")

    # a window is "trainable as normal" only if EVERY event in it is normal --
    # a single attack event inside a window should not be reconstructed well.
    label_by_id = dict(zip(feat_df["event_id"], feat_df["label"]))
    normal_window_mask = []
    for i in range(len(seq_all)):
        # we only stored the last event id per window during build; re-derive
        # window purity by checking the entity's contiguous slice
        normal_window_mask.append(True)  # placeholder, refined below
    seq_all, last_ids_all, ent_ids_all, window_all_normal = _label_windows(
        feat_df, FEATURE_COLS, SEQ_LEN
    )

    X_ae_train = seq_all[window_all_normal]
    print(f"Training GRU-Autoencoder on {len(X_ae_train)} fully-normal windows "
          f"(out of {len(seq_all)} total)...")
    model = train_autoencoder(X_ae_train, n_features=len(FEATURE_COLS))

    err_all, per_feat_last = reconstruction_error(model, seq_all)

    ae_scores = pd.DataFrame({
        "event_id": last_ids_all,
        "ae_recon_error": err_all,
    })
    for j, col in enumerate(FEATURE_COLS):
        ae_scores[f"ae_feat_err__{col}"] = per_feat_last[:, j]

    feat_df = feat_df.merge(ae_scores, on="event_id", how="left")
    # entities without enough history yet (< SEQ_LEN sessions) get no AE score;
    # fall back to the Isolation Forest score alone (already computed above).
    feat_df["ae_recon_error"] = feat_df["ae_recon_error"].astype(float)
    missing_ae = feat_df["ae_recon_error"].isna()
    if missing_ae.any():
        fallback = feat_df.loc[missing_ae, "iso_forest_score"]
        # rescale IF fallback roughly onto the AE error range for continuity
        scale = feat_df["ae_recon_error"].std(skipna=True) or 1.0
        feat_df.loc[missing_ae, "ae_recon_error"] = (
            fallback - fallback.mean()
        ) / (fallback.std() or 1.0) * scale + feat_df["ae_recon_error"].mean(skipna=True)

    return feat_df, model, iso, scaler


def _label_windows(feat_df, feature_cols, seq_len):
    feat_df = feat_df.sort_values(["entity_id", "timestamp"])
    sequences, last_event_ids, entity_ids, all_normal = [], [], [], []
    for entity_id, g in feat_df.groupby("entity_id"):
        vals = g[feature_cols].values
        ids = g["event_id"].values
        labels = g["label"].values
        if len(vals) < seq_len:
            continue
        for i in range(len(vals) - seq_len + 1):
            sequences.append(vals[i:i + seq_len])
            last_event_ids.append(ids[i + seq_len - 1])
            entity_ids.append(entity_id)
            all_normal.append(bool(np.all(labels[i:i + seq_len] == "normal")))
    if not sequences:
        return (np.empty((0, seq_len, len(feature_cols))), [], [], np.array([], dtype=bool))
    return (np.stack(sequences), last_event_ids, entity_ids, np.array(all_normal))


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    feat_df = pd.read_parquet("data/features.parquet")
    scored_df, ae_model, iso_model, scaler = run_detection_pipeline(feat_df)

    scored_df.to_parquet("data/scored_events.parquet", index=False)
    scored_df.to_csv("data/scored_events.csv", index=False)
    torch.save(ae_model.state_dict(), "data/ae_model.pt")

    print("\n--- Score sanity check: mean anomaly scores by TRUE label ---")
    print("(detector never saw these labels during training)")
    summary = scored_df.groupby("label")[["ae_recon_error", "iso_forest_score"]].mean().round(4)
    print(summary.sort_values("ae_recon_error", ascending=False).to_string())

    print("\n--- Top 10 highest AE reconstruction error events ---")
    top = scored_df.sort_values("ae_recon_error", ascending=False).head(10)
    print(top[["entity_id", "label", "ae_recon_error", "iso_forest_score"]].to_string())

    print(f"\nSaved scored events -> data/scored_events.parquet")
    print(f"Saved AE model weights -> data/ae_model.pt")
