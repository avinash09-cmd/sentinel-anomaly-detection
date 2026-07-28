"""
data_generator.py
==================
Synthetic behavioral-telemetry generator for the AI-Powered Behavioral Anomaly
Detection prototype.

Design rationale (why this approach):
- We model each entity (user / service_account / edge_device) with a persistent
  "behavioral signature" (habitual hours, home geo, typical resources, typical
  device) because real detectors work by comparing NEW behavior to an entity's
  OWN history -- not a global average. Synthetic data must therefore be
  generated the same way, or downstream models would learn a shortcut that
  doesn't transfer to real telemetry.
- Attacks are injected as structural violations of a specific dimension of the
  signature (geo, device, resource-set, timing, volume) so that each attack
  type is separable in principle -- this lets us later prove the classifier
  is learning real signal, not label leakage.
- Two output modes (batch labeled vs. streaming unlabeled) mirror the real
  deployment split: offline model training/eval vs. online scoring.
"""

import argparse
import json
import math
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# A fixed pool of "cities" with lat/lon so we can compute geo-velocity
# (km / hour) for impossible-travel detection. Using a fixed pool (rather than
# fully random faker geos) keeps distances physically meaningful and reusable
# across profiler/detector features.
CITIES = [
    ("New York", 40.7128, -74.0060), ("London", 51.5072, -0.1276),
    ("Frankfurt", 50.1109, 8.6821), ("Singapore", 1.3521, 103.8198),
    ("Sydney", -33.8688, 151.2093), ("Sao Paulo", -23.5505, -46.6333),
    ("Mumbai", 19.0760, 72.8777), ("Tokyo", 35.6762, 139.6503),
    ("Johannesburg", -26.2041, 28.0473), ("Toronto", 43.6532, -79.3832),
    ("Dublin", 53.3498, -6.2603), ("Dubai", 25.2048, 55.2708),
    ("Chicago", 41.8781, -87.6298), ("Berlin", 52.5200, 13.4050),
    ("Seoul", 37.5665, 126.9780), ("Austin", 30.2672, -97.7431),
]

AUTH_METHODS = ["password", "sso_saml", "mfa_push", "api_key", "cert_based"]

RESOURCE_POOL = {
    "user": ["hr_portal", "email", "shared_drive_finance", "shared_drive_eng",
             "crm", "vpn_gateway", "wiki", "ticketing_system", "payroll_admin",
             "code_repo", "billing_console", "customer_db"],
    "service_account": ["billing_api", "customer_db", "backup_service",
                         "message_queue", "ci_cd_pipeline", "secrets_vault",
                         "log_aggregator", "monitoring_api"],
    "edge_device": ["pos_gateway", "iot_hub_telemetry", "firmware_update_svc",
                    "edge_config_api", "camera_feed_relay", "sensor_ingest"],
}

COMMAND_POOL = {
    "user": ["view_dashboard", "download_report", "edit_record", "search",
              "export_csv", "send_email", "open_ticket", "git_pull", "git_push"],
    "service_account": ["read_batch", "write_batch", "rotate_key", "healthcheck",
                        "sync_job", "backup_run", "publish_msg", "consume_msg"],
    "edge_device": ["heartbeat", "firmware_check", "telemetry_push",
                    "config_pull", "reboot", "sensor_read"],
}

ATTACK_TYPES = [
    "credential_misuse", "lateral_movement", "brute_force",
    "impossible_travel", "device_spoofing", "low_and_slow_exfiltration",
    "insider_drift",
]

RNG = np.random.default_rng(42)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def new_fingerprint():
    return "fp_" + "".join(random.choices(string.hexdigits.lower(), k=12))


def new_ip():
    return fake.ipv4_public()


# ---------------------------------------------------------------------------
# Entity behavioral signature
# ---------------------------------------------------------------------------

@dataclass
class EntitySignature:
    entity_id: str
    entity_type: str
    role_group: str                      # peer group for cold-start fallback
    home_city: tuple                     # (name, lat, lon)
    login_hour_mean: float                # habitual hour-of-day (0-24, circular)
    login_hour_std: float
    active_days: list                    # weekdays typically active, 0=Mon
    typical_resources: list              # subset of RESOURCE_POOL[type]
    typical_devices: list                # 1-2 device fingerprints
    session_duration_mean: float          # seconds
    session_duration_std: float
    typical_bytes_mean: float             # bytes transferred per session
    typical_bytes_std: float
    preferred_auth: str
    created_at: datetime

    def sample_hour(self):
        h = RNG.normal(self.login_hour_mean, self.login_hour_std)
        return float(h % 24)

    def sample_weekday_ok(self, dt):
        # weekends allowed with small probability for humans, always ok for
        # service accounts / devices (they don't take weekends off)
        if self.entity_type != "user":
            return True
        return dt.weekday() in self.active_days or RNG.random() < 0.05


def make_entity_signature(idx, entity_type, start_date):
    role_groups = {
        "user": ["engineering", "finance", "sales", "hr", "support"],
        "service_account": ["data_pipeline", "billing_infra", "ci_infra"],
        "edge_device": ["retail_pos", "iot_fleet", "gateway_fleet"],
    }
    role = random.choice(role_groups[entity_type])
    city = random.choice(CITIES)
    n_res = {"user": 4, "service_account": 3, "edge_device": 2}[entity_type]
    resources = random.sample(RESOURCE_POOL[entity_type],
                               k=min(n_res, len(RESOURCE_POOL[entity_type])))
    n_dev = 1 if entity_type != "user" else random.choice([1, 1, 2])
    devices = [new_fingerprint() for _ in range(n_dev)]

    return EntitySignature(
        entity_id=f"{entity_type}_{idx:04d}",
        entity_type=entity_type,
        role_group=role,
        home_city=city,
        login_hour_mean=RNG.uniform(0, 24) if entity_type != "user" else RNG.normal(13, 3) % 24,
        login_hour_std=1.5 if entity_type == "user" else 3.0,
        active_days=sorted(random.sample(range(7), k=5)) if entity_type == "user" else list(range(7)),
        typical_resources=resources,
        typical_devices=devices,
        session_duration_mean=RNG.uniform(180, 1800),
        session_duration_std=RNG.uniform(30, 300),
        typical_bytes_mean=RNG.uniform(5_000, 500_000),
        typical_bytes_std=RNG.uniform(1_000, 50_000),
        preferred_auth=random.choice(AUTH_METHODS),
        created_at=start_date - timedelta(days=random.randint(30, 400)),
    )


# ---------------------------------------------------------------------------
# Normal session generation
# ---------------------------------------------------------------------------

def gen_normal_event(sig: EntitySignature, day: datetime, event_id, force_new=False):
    """One normal session drawn from the entity's own signature + noise."""
    hour = sig.sample_hour()
    ts = day.replace(hour=0, minute=0, second=0) + timedelta(hours=hour)

    city_name, lat, lon = sig.home_city
    # small home-radius jitter (mobile / vpn exit variance), not a new city
    lat_j = lat + RNG.normal(0, 0.05)
    lon_j = lon + RNG.normal(0, 0.05)

    resource = random.choice(sig.typical_resources) if RNG.random() > 0.1 \
        else random.choice(RESOURCE_POOL[sig.entity_type])  # rare benign novelty
    device = random.choice(sig.typical_devices)
    duration = max(5, RNG.normal(sig.session_duration_mean, sig.session_duration_std))
    n_cmds = max(1, int(RNG.poisson(4)))
    commands = ";".join(random.choices(COMMAND_POOL[sig.entity_type], k=n_cmds))
    bytes_out = max(0, RNG.normal(sig.typical_bytes_mean, sig.typical_bytes_std))

    return {
        "event_id": event_id,
        "entity_id": sig.entity_id,
        "entity_type": sig.entity_type,
        "role_group": sig.role_group,
        "timestamp": ts,
        "source_ip": new_ip(),
        "geo_city": city_name,
        "geo_lat": round(lat_j, 4),
        "geo_lon": round(lon_j, 4),
        "resource_accessed": resource,
        "auth_method": sig.preferred_auth,
        "auth_result": "success" if RNG.random() > 0.02 else "fail",
        "session_duration": round(duration, 1),
        "command_sequence": commands,
        "device_fingerprint": device,
        "bytes_transferred": round(bytes_out, 1),
        "label": "normal",
    }


# ---------------------------------------------------------------------------
# Attack injection -- each returns a LIST of events (some attacks span several)
# ---------------------------------------------------------------------------

def attack_credential_misuse(sig, day, event_id_fn):
    """Stolen creds used from an unfamiliar geo + unfamiliar device, but auth
    succeeds outright (no brute force) -- e.g. phished token replay."""
    base = gen_normal_event(sig, day, event_id_fn())
    foreign_city = random.choice([c for c in CITIES if c[0] != sig.home_city[0]])
    base.update({
        "geo_city": foreign_city[0],
        "geo_lat": foreign_city[1] + RNG.normal(0, 0.1),
        "geo_lon": foreign_city[2] + RNG.normal(0, 0.1),
        "device_fingerprint": new_fingerprint(),
        "resource_accessed": random.choice(RESOURCE_POOL[sig.entity_type]),
        "auth_method": random.choice(AUTH_METHODS),
        "label": "credential_misuse",
    })
    return [base]


def attack_lateral_movement(sig, day, event_id_fn):
    """After a plausible entry, rapid-fire access across MANY resources the
    entity has never/rarely touched -- reconnaissance / pivoting signature."""
    events = []
    start_hour = sig.sample_hour()
    ts0 = day.replace(hour=0, minute=0, second=0) + timedelta(hours=start_hour)
    all_res = RESOURCE_POOL[sig.entity_type] + \
        RESOURCE_POOL["service_account"] if sig.entity_type != "service_account" else RESOURCE_POOL[sig.entity_type]
    touch = random.sample(all_res, k=min(6, len(all_res)))
    for i, res in enumerate(touch):
        e = gen_normal_event(sig, day, event_id_fn())
        e["timestamp"] = ts0 + timedelta(minutes=i * random.uniform(1, 4))
        e["resource_accessed"] = res
        e["session_duration"] = round(RNG.uniform(5, 40), 1)  # quick, mechanical
        e["command_sequence"] = "enumerate;access;pivot"
        e["label"] = "lateral_movement"
        events.append(e)
    return events


def attack_brute_force(sig, day, event_id_fn):
    """Burst of auth failures in a tight window, then (usually) one success."""
    events = []
    start_hour = sig.sample_hour()
    ts0 = day.replace(hour=0, minute=0, second=0) + timedelta(hours=start_hour)
    n_fail = random.randint(6, 25)
    ip = new_ip()
    for i in range(n_fail):
        e = gen_normal_event(sig, day, event_id_fn())
        e["timestamp"] = ts0 + timedelta(seconds=i * random.uniform(2, 8))
        e["source_ip"] = ip
        e["auth_result"] = "fail"
        e["auth_method"] = random.choice(AUTH_METHODS)
        e["session_duration"] = 0.0
        e["label"] = "brute_force"
        events.append(e)
    if random.random() < 0.6:
        e = gen_normal_event(sig, day, event_id_fn())
        e["timestamp"] = ts0 + timedelta(seconds=n_fail * 5 + 10)
        e["source_ip"] = ip
        e["auth_result"] = "success"
        e["label"] = "brute_force"
        events.append(e)
    return events


def attack_impossible_travel(sig, day, event_id_fn):
    """Two sessions, two distant geos, time delta too short for real travel."""
    e1 = gen_normal_event(sig, day, event_id_fn())
    foreign_city = random.choice([c for c in CITIES if c[0] != sig.home_city[0]])
    dist_km = haversine_km(sig.home_city[1], sig.home_city[2], foreign_city[1], foreign_city[2])
    # commercial flight ~800km/h; pick a delta that implies >1500km/h
    min_realistic_hours = dist_km / 800
    delta_hours = min_realistic_hours * random.uniform(0.05, 0.3)
    e2 = gen_normal_event(sig, day, event_id_fn())
    e2["timestamp"] = e1["timestamp"] + timedelta(hours=max(0.05, delta_hours))
    e2["geo_city"] = foreign_city[0]
    e2["geo_lat"] = foreign_city[1]
    e2["geo_lon"] = foreign_city[2]
    e2["device_fingerprint"] = random.choice(sig.typical_devices)  # same device -> impossible
    e1["label"] = e2["label"] = "impossible_travel"
    return [e1, e2]


def attack_device_spoofing(sig, day, event_id_fn):
    """Right geo, right hours, right resources -- but an entirely new,
    never-seen device fingerprint. Isolates the device signal from geo/time."""
    e = gen_normal_event(sig, day, event_id_fn())
    e["device_fingerprint"] = new_fingerprint()
    e["label"] = "device_spoofing"
    return [e]


def attack_low_and_slow_exfil(sig, day, event_id_fn):
    """Several sessions across the day with a gentle, sustained bump in bytes
    transferred -- designed to stay under any single-session threshold."""
    events = []
    n = random.randint(3, 6)
    for i in range(n):
        e = gen_normal_event(sig, day, event_id_fn())
        e["timestamp"] = e["timestamp"] + timedelta(hours=i * random.uniform(1, 3))
        e["bytes_transferred"] = round(sig.typical_bytes_mean * random.uniform(1.6, 2.4)
                                        + sig.typical_bytes_std, 1)
        e["resource_accessed"] = random.choice(sig.typical_resources)
        e["command_sequence"] = "export_csv;export_csv;compress"
        e["label"] = "low_and_slow_exfiltration"
        events.append(e)
    return events


def attack_insider_drift(sig, day, event_id_fn):
    """A single day sampled from a signature that has been slowly, gradually
    nudged away from baseline over preceding weeks (see inject_all_attacks) --
    at any one point it looks *almost* normal, which is what makes it hard."""
    e = gen_normal_event(sig, day, event_id_fn())
    # slightly-off hour, slightly novel resource, slightly larger export --
    # individually subtle, jointly a drift signature
    e["timestamp"] = e["timestamp"] + timedelta(hours=random.uniform(2, 5))
    e["resource_accessed"] = random.choice(RESOURCE_POOL[sig.entity_type])
    e["bytes_transferred"] = round(e["bytes_transferred"] * random.uniform(1.3, 1.8), 1)
    e["command_sequence"] = e["command_sequence"] + ";export_csv"
    e["label"] = "insider_drift"
    return [e]


ATTACK_FN = {
    "credential_misuse": attack_credential_misuse,
    "lateral_movement": attack_lateral_movement,
    "brute_force": attack_brute_force,
    "impossible_travel": attack_impossible_travel,
    "device_spoofing": attack_device_spoofing,
    "low_and_slow_exfiltration": attack_low_and_slow_exfil,
    "insider_drift": attack_insider_drift,
}


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_entities=60, n_days=30, sessions_per_day_lambda=3.0,
                      attack_rate=0.02, start_date=None, seed=42):
    """
    attack_rate: fraction of ENTITY-DAYS that get an injected attack (spread
    across the 7 types). Kept configurable 0.5%-3% per the brief.
    """
    random.seed(seed)
    RNG_local = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2026, 6, 1)

    entity_types = ["user"] * int(n_entities * 0.7) + \
                    ["service_account"] * int(n_entities * 0.2) + \
                    ["edge_device"] * (n_entities - int(n_entities * 0.7) - int(n_entities * 0.2))
    signatures = [make_entity_signature(i, t, start_date) for i, t in enumerate(entity_types)]

    counter = {"n": 0}
    def next_id():
        counter["n"] += 1
        return f"evt_{counter['n']:08d}"

    rows = []
    attack_counts = {k: 0 for k in ATTACK_TYPES}

    for day_offset in range(n_days):
        day = start_date + timedelta(days=day_offset)
        for sig in signatures:
            if not sig.sample_weekday_ok(day):
                continue
            n_sessions = max(0, int(RNG_local.poisson(sessions_per_day_lambda)))

            # decide if this entity-day is attacked
            is_attacked = RNG_local.random() < attack_rate
            attack_type = random.choice(ATTACK_TYPES) if is_attacked else None

            for _ in range(n_sessions):
                rows.append(gen_normal_event(sig, day, next_id()))

            if attack_type:
                rows.extend(ATTACK_FN[attack_type](sig, day, next_id))
                attack_counts[attack_type] += 1

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    meta = {
        "n_entities": n_entities,
        "n_days": n_days,
        "attack_rate": attack_rate,
        "attack_counts": attack_counts,
        "total_events": len(df),
    }
    return df, signatures, meta


def stream_events(df, speed_multiplier=None):
    """
    Generator that yields UNLABELED events in timestamp order, simulating a
    live feed. If speed_multiplier is set, sleeps proportionally between
    events (real-time feel for the dashboard); if None, yields instantly
    (for offline consumption / testing).
    """
    stream_df = df.drop(columns=["label"]).sort_values("timestamp")
    prev_ts = None
    for _, row in stream_df.iterrows():
        if speed_multiplier and prev_ts is not None:
            real_delta = (row["timestamp"] - prev_ts).total_seconds()
            time.sleep(min(real_delta / speed_multiplier, 2.0))
        prev_ts = row["timestamp"]
        yield row.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic behavioral telemetry")
    ap.add_argument("--entities", type=int, default=60)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--attack-rate", type=float, default=0.02)
    ap.add_argument("--out", type=str, default="data/events.parquet")
    ap.add_argument("--csv-out", type=str, default="data/events.csv")
    args = ap.parse_args()

    import os
    os.makedirs("data", exist_ok=True)

    df, signatures, meta = generate_dataset(
        n_entities=args.entities, n_days=args.days, attack_rate=args.attack_rate
    )

    df.to_parquet(args.out, index=False)
    df.to_csv(args.csv_out, index=False)

    print(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved {len(df)} events -> {args.out} / {args.csv_out}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nSample rows:")
    print(df.sample(5, random_state=1).to_string())
