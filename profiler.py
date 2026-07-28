"""
profiler.py
===========
Baseline behavioral profiler: builds and maintains a per-entity statistical
profile of "normal" behavior, with:
  - exponential decay so profiles adapt to legitimate drift (constraint #3)
  - a peer-group fallback for cold-start entities with < COLD_START_MIN_SESSIONS
    of personal history (constraint #5)

This module is deliberately independent of the deep model (detector.py) --
its outputs (per-event deviation z-scores / flags) become engineered features
for both the Isolation Forest baseline and the GBM classifier, and are also
what a SOC analyst reads directly in plain English.
"""

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Below this many personal sessions, we blend in the peer-group profile.
# Chosen because ~15 sessions gives a stable-enough hour/geo/resource
# distribution for the entity types in this dataset (few sessions/day).
COLD_START_MIN_SESSIONS = 15

# Exponential decay half-life for the rolling profile, in sessions. A ~40
# session half-life adapts to genuine role changes within a few weeks while
# not flapping session-to-session.
DECAY_HALF_LIFE = 40.0
DECAY_LAMBDA = math.log(2) / DECAY_HALF_LIFE


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class EWMAStats:
    """Exponentially-weighted mean/variance for a scalar signal, used for
    login-hour and session-duration drift instead of a fixed-window mean/std
    (fixed windows either freeze on old behavior or forget too fast)."""
    def __init__(self):
        self.mean = None
        self.var = 1.0
        self.n = 0

    def update(self, x, alpha=None):
        alpha = alpha if alpha is not None else (1 - math.exp(-DECAY_LAMBDA))
        if self.mean is None:
            self.mean = x
            self.var = 1.0
        else:
            diff = x - self.mean
            incr = alpha * diff
            self.mean += incr
            self.var = (1 - alpha) * (self.var + alpha * diff * diff)
        self.n += 1

    def zscore(self, x):
        std = math.sqrt(self.var) if self.var > 1e-6 else 1.0
        return (x - self.mean) / std if self.mean is not None else 0.0


class ResourceHistogram:
    """Decayed access-count histogram -> gives a 'novelty score' for a
    resource (1 - normalized frequency), robust to slow legitimate drift."""
    def __init__(self):
        self.counts = defaultdict(float)

    def update(self, resource, decay=0.99):
        for k in list(self.counts.keys()):
            self.counts[k] *= decay
        self.counts[resource] += 1.0

    def novelty(self, resource):
        total = sum(self.counts.values())
        if total < 1e-6:
            return 1.0  # never seen anything -> fully novel
        freq = self.counts.get(resource, 0.0) / total
        return 1.0 - freq


class EntityProfile:
    def __init__(self, entity_id, entity_type, role_group):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.role_group = role_group
        self.n_sessions = 0
        self.hour_stats = EWMAStats()
        self.duration_stats = EWMAStats()
        self.bytes_stats = EWMAStats()
        self.resource_hist = ResourceHistogram()
        self.known_devices = defaultdict(float)   # fingerprint -> decayed count
        self.geo_centroid = None                  # (lat, lon), EWMA
        self.geo_radius = 50.0                     # km, EWMA of distance from centroid
        self.last_event = None                     # (timestamp, lat, lon) for geo-velocity
        self.recent_auth_fails = deque(maxlen=50)   # timestamps, for burst detection

    def is_cold_start(self):
        return self.n_sessions < COLD_START_MIN_SESSIONS

    def update(self, event):
        hour = pd.Timestamp(event["timestamp"]).hour + pd.Timestamp(event["timestamp"]).minute / 60
        self.hour_stats.update(hour)
        self.duration_stats.update(event["session_duration"])
        self.bytes_stats.update(event["bytes_transferred"])
        self.resource_hist.update(event["resource_accessed"])
        for k in list(self.known_devices.keys()):
            self.known_devices[k] *= 0.995
        self.known_devices[event["device_fingerprint"]] += 1.0

        lat, lon = event["geo_lat"], event["geo_lon"]
        if self.geo_centroid is None:
            self.geo_centroid = (lat, lon)
        else:
            alpha = 1 - math.exp(-DECAY_LAMBDA)
            self.geo_centroid = (
                self.geo_centroid[0] + alpha * (lat - self.geo_centroid[0]),
                self.geo_centroid[1] + alpha * (lon - self.geo_centroid[1]),
            )
        dist = haversine_km(self.geo_centroid[0], self.geo_centroid[1], lat, lon)
        alpha_r = 1 - math.exp(-DECAY_LAMBDA)
        self.geo_radius = max(5.0, self.geo_radius + alpha_r * (dist - self.geo_radius))

        if event["auth_result"] == "fail":
            self.recent_auth_fails.append(pd.Timestamp(event["timestamp"]))

        self.n_sessions += 1
        self.last_event = (pd.Timestamp(event["timestamp"]), lat, lon)


class PeerGroupProfile:
    """Aggregated profile across all entities sharing a role_group, used to
    score cold-start entities before they have enough personal history."""
    def __init__(self, role_group):
        self.role_group = role_group
        self.hour_stats = EWMAStats()
        self.duration_stats = EWMAStats()
        self.bytes_stats = EWMAStats()
        self.resource_hist = ResourceHistogram()
        self.n_sessions = 0

    def update(self, event):
        hour = pd.Timestamp(event["timestamp"]).hour + pd.Timestamp(event["timestamp"]).minute / 60
        self.hour_stats.update(hour, alpha=0.01)  # slower decay: peer group is a stable reference
        self.duration_stats.update(event["session_duration"], alpha=0.01)
        self.bytes_stats.update(event["bytes_transferred"], alpha=0.01)
        self.resource_hist.update(event["resource_accessed"], decay=0.999)
        self.n_sessions += 1


class BehavioralProfiler:
    """Owns all per-entity and per-peer-group profiles, processes events in
    timestamp order (as in a real stream), and emits engineered deviation
    features for each event BEFORE updating its own state with that event
    (critical: prevents label leakage / lookahead)."""

    def __init__(self):
        self.entities = {}
        self.peer_groups = {}

    def _get_entity(self, event):
        eid = event["entity_id"]
        if eid not in self.entities:
            self.entities[eid] = EntityProfile(eid, event["entity_type"], event["role_group"])
        return self.entities[eid]

    def _get_peer(self, role_group):
        if role_group not in self.peer_groups:
            self.peer_groups[role_group] = PeerGroupProfile(role_group)
        return self.peer_groups[role_group]

    def score_event(self, event):
        """Return a dict of engineered features describing how much this
        event deviates from the entity's (or peer group's, if cold-start)
        profile. Called BEFORE .observe(event)."""
        ent = self._get_entity(event)
        peer = self._get_peer(event["role_group"])
        cold_start = ent.is_cold_start()
        # blend weight: 0 = fully personal, 1 = fully peer. Linearly ramps
        # personal trust in from 0 -> COLD_START_MIN_SESSIONS sessions.
        blend = 0.0 if not cold_start else 1.0 - (ent.n_sessions / COLD_START_MIN_SESSIONS)

        hour = pd.Timestamp(event["timestamp"]).hour + pd.Timestamp(event["timestamp"]).minute / 60

        # --- time-of-day deviation ---
        z_personal = ent.hour_stats.zscore(hour) if ent.hour_stats.mean is not None else 0.0
        z_peer = peer.hour_stats.zscore(hour) if peer.hour_stats.mean is not None else 0.0
        hour_z = blend * z_peer + (1 - blend) * z_personal

        # --- geo-velocity (km/h) vs last known event for this entity ---
        geo_velocity = 0.0
        geo_dev_km = 0.0
        if ent.last_event is not None:
            last_ts, last_lat, last_lon = ent.last_event
            cur_ts = pd.Timestamp(event["timestamp"])
            dt_hours = max((cur_ts - last_ts).total_seconds() / 3600.0, 1e-4)
            dist = haversine_km(last_lat, last_lon, event["geo_lat"], event["geo_lon"])
            geo_velocity = dist / dt_hours
        if ent.geo_centroid is not None:
            geo_dev_km = haversine_km(ent.geo_centroid[0], ent.geo_centroid[1],
                                       event["geo_lat"], event["geo_lon"])
            geo_dev_km = max(0.0, geo_dev_km - ent.geo_radius)  # excess beyond typical radius

        # --- resource novelty ---
        res_novelty_personal = ent.resource_hist.novelty(event["resource_accessed"])
        res_novelty_peer = peer.resource_hist.novelty(event["resource_accessed"])
        resource_novelty = blend * res_novelty_peer + (1 - blend) * res_novelty_personal

        # --- device mismatch ---
        device_known_personal = ent.known_devices.get(event["device_fingerprint"], 0.0)
        device_mismatch = 1.0 if (not cold_start and device_known_personal < 0.5) else \
            (0.5 if cold_start else 0.0)

        # --- session-duration deviation ---
        dur_z_personal = ent.duration_stats.zscore(event["session_duration"]) if ent.duration_stats.mean is not None else 0.0
        dur_z_peer = peer.duration_stats.zscore(event["session_duration"]) if peer.duration_stats.mean is not None else 0.0
        duration_z = blend * dur_z_peer + (1 - blend) * dur_z_personal

        # --- bytes-transferred deviation ---
        bytes_z_personal = ent.bytes_stats.zscore(event["bytes_transferred"]) if ent.bytes_stats.mean is not None else 0.0
        bytes_z_peer = peer.bytes_stats.zscore(event["bytes_transferred"]) if peer.bytes_stats.mean is not None else 0.0
        bytes_z = blend * bytes_z_peer + (1 - blend) * bytes_z_personal

        # --- auth-failure burst rate (last 5 min) ---
        cur_ts = pd.Timestamp(event["timestamp"])
        recent_fails = sum(1 for t in ent.recent_auth_fails if (cur_ts - t).total_seconds() < 300)
        if event["auth_result"] == "fail":
            recent_fails += 1
        auth_fail_burst = recent_fails

        features = {
            "hour_deviation_z": float(hour_z),
            "geo_velocity_kmh": float(geo_velocity),
            "geo_deviation_km": float(geo_dev_km),
            "resource_novelty": float(resource_novelty),
            "device_mismatch": float(device_mismatch),
            "duration_deviation_z": float(duration_z),
            "bytes_deviation_z": float(bytes_z),
            "auth_fail_burst": float(auth_fail_burst),
            "is_cold_start": bool(cold_start),
            "personal_session_count": int(ent.n_sessions),
        }
        return features

    def observe(self, event):
        """Update entity + peer-group state with this event (call AFTER
        score_event, in timestamp order)."""
        ent = self._get_entity(event)
        peer = self._get_peer(event["role_group"])
        ent.update(event)
        peer.update(event)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Stream a labeled/unlabeled dataframe through the profiler in timestamp
    order and return the engineered feature table (one row per event),
    correctly avoiding lookahead by scoring before observing."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    profiler = BehavioralProfiler()
    feature_rows = []
    for _, event in df.iterrows():
        ev = event.to_dict()
        feats = profiler.score_event(ev)
        feats["event_id"] = ev["event_id"]
        feature_rows.append(feats)
        profiler.observe(ev)
    feat_df = pd.DataFrame(feature_rows)
    out = df.merge(feat_df, on="event_id", how="left")
    return out, profiler


if __name__ == "__main__":
    df = pd.read_parquet("data/events.parquet")
    print(f"Loaded {len(df)} events. Building behavioral profiles + features (streaming order)...")
    feat_df, profiler = build_feature_table(df)

    feat_df.to_parquet("data/features.parquet", index=False)
    feat_df.to_csv("data/features.csv", index=False)

    print(f"\nBuilt profiles for {len(profiler.entities)} entities across "
          f"{len(profiler.peer_groups)} peer groups.")
    cold = sum(1 for e in profiler.entities.values() if e.is_cold_start())
    print(f"Entities still in cold-start (<{COLD_START_MIN_SESSIONS} sessions) "
          f"at end of stream: {cold}")

    print("\nFeature summary by label (mean values -- sanity check that attacks "
          "actually deviate more than normal traffic):")
    cols = ["hour_deviation_z", "geo_velocity_kmh", "geo_deviation_km",
            "resource_novelty", "device_mismatch", "duration_deviation_z",
            "bytes_deviation_z", "auth_fail_burst"]
    print(feat_df.groupby("label")[cols].mean().round(2).to_string())

    print("\nSample feature rows:")
    print(feat_df[["entity_id", "label"] + cols].sample(8, random_state=1).to_string())
