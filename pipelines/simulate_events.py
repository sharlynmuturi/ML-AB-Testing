# pipelines/simulate_events.py
"""
Simulates event logs for an ML-powered A/B test system.
Each run represents new users arriving into the system.
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
np.random.seed(42)

# Load config
with open("pipelines/config.json") as f:
    CONFIG = json.load(f)

# Convert paths to Path objects
for k, v in CONFIG["paths"].items():
    CONFIG["paths"][k] = Path(v)

# Ensure checkpoint directories exist
for path in [CONFIG["paths"]["checkpoint_ts"].parent, CONFIG["paths"]["run_counter"].parent, CONFIG["paths"]["raw_out"].parent]:
    path.mkdir(parents=True, exist_ok=True)


def get_start_time():
    """Determine start time based on last simulation timestamp to ensure new runs don’t overlap with old data."""
    checkpoint = CONFIG["paths"]["checkpoint_ts"]
    if checkpoint.exists():
        last_ts = pd.to_datetime(json.load(open(checkpoint))["last_ts"])
        return last_ts + timedelta(seconds=5)
    return datetime.now()


def get_run_id():
    """Get the next run ID and increment the counter."""
    counter_path = CONFIG["paths"]["run_counter"]
    if counter_path.exists():
        run_id = int(counter_path.read_text())
    else:
        run_id = 0
    counter_path.write_text(str(run_id + 1))
    return run_id


def generate_timestamp(base_time, max_minutes=60):
    """Generate random timestamp within max_minutes of base_time."""
    return base_time + timedelta(minutes=np.random.randint(0, max_minutes))


def assign_variant():
    """Randomly assign user to a variant based on variant split."""
    variants = list(CONFIG["variant_split"].keys())
    probs = list(CONFIG["variant_split"].values())
    return np.random.choice(variants, p=probs)


# Main Simulation
def simulate_events(n_users=None, start_time=None, run_id=None):
    n_users = n_users or CONFIG["n_users"]
    start_time = start_time or get_start_time()
    run_id = run_id if run_id is not None else get_run_id()

    users = [f"user_{run_id}_{i}" for i in range(n_users)]
    records = []

    for user_id in users:
        variant = assign_variant()
        model_info = CONFIG["model_config"][variant]

        # Variant assignment
        records.append({
            "event_id": str(uuid.uuid4()),
            "event_type": "variant_assignment",
            "timestamp": generate_timestamp(start_time),
            "user_id": user_id,
            "experiment_id": CONFIG["experiment_id"],
            "variant": variant,
            "model_version": model_info["model_version"],
            "prediction_score": None,
            "latency_ms": None,
            "clicked": None
        })

        # Model inference
        prediction_score = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        latency = max(5, int(np.random.normal(model_info["latency_mean"], 8)))
        records.append({
            "event_id": str(uuid.uuid4()),
            "event_type": "model_inference",
            "timestamp": generate_timestamp(start_time),
            "user_id": user_id,
            "experiment_id": CONFIG["experiment_id"],
            "variant": variant,
            "model_version": model_info["model_version"],
            "prediction_score": prediction_score,
            "latency_ms": latency,
            "clicked": None
        })

        # User response
        clicked = np.random.binomial(1, model_info["ctr"])
        records.append({
            "event_id": str(uuid.uuid4()),
            "event_type": "user_response",
            "timestamp": generate_timestamp(start_time),
            "user_id": user_id,
            "experiment_id": CONFIG["experiment_id"],
            "variant": variant,
            "model_version": model_info["model_version"],
            "prediction_score": None,
            "latency_ms": None,
            "clicked": clicked
        })

    events_df = pd.DataFrame(records)
    return events_df, run_id


def save_events(events_df):
    """Append events to CSV file."""
    raw_path = CONFIG["paths"]["raw_out"]

    if raw_path.exists():
        events_df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        events_df.to_csv(raw_path, index=False)

    logging.info(f"Simulation complete: {len(events_df)//3} users added")
    logging.info(events_df.head())


if __name__ == "__main__":
    events, run_id = simulate_events()
    save_events(events)