# pipelines/incremental_aggregate.py
"""
Incremental aggregation of ML experiment event logs.
Maintains cumulative metrics per experiment variant.
Process only new data since last run
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load config
with open("pipelines/config.json") as f:
    CONFIG = json.load(f)

# Convert paths to Path objects
for k, v in CONFIG["paths"].items():
    CONFIG["paths"][k] = Path(v)

# Ensure directories exist
for p in ["data/processed", "data/checkpoints"]:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_last_timestamp():
    """Load timestamp checkpoint for incremental processing."""
    path = CONFIG["paths"]["checkpoint_ts"]
    if path.exists():
        return pd.to_datetime(json.load(open(path))["last_ts"])
    return None


def load_events(last_ts=None):
    """Load raw events and filter new ones based on last timestamp."""
    df = pd.read_csv(CONFIG["paths"]["raw_out"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if last_ts is not None:
        df = df[df["timestamp"] > last_ts]
    return df


def load_seen_users():
    """Load previously seen users for deduplication."""
    path = CONFIG["paths"]["seen_users"]
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["experiment_id", "variant", "user_id"])


def deduplicate_users(df, seen_users):
    """Return new users only and update seen users file."""
    assignments = df[df["event_type"] == "variant_assignment"][["experiment_id", "variant", "user_id"]].drop_duplicates()
    new_users = assignments.merge(seen_users, on=["experiment_id", "variant", "user_id"], how="left", indicator=True)
    new_users = new_users[new_users["_merge"] == "left_only"][["experiment_id", "variant", "user_id"]]

    # Update global seen users file
    updated_seen = pd.concat([seen_users, new_users], ignore_index=True)
    updated_seen.to_csv(CONFIG["paths"]["seen_users"], index=False)
    return new_users


def compute_metrics(df, new_users):
    """Compute incremental and cumulative metrics per variant."""
    # New user counts
    new_user_counts = new_users.groupby(["experiment_id", "variant"]).size().reset_index(name="new_users")

    # Incremental clicks & impressions
    responses = df[df["event_type"] == "user_response"]
    clicks = (
        responses.groupby(["experiment_id", "variant"])
        .agg(new_impressions=("user_id", "count"), new_clicks=("clicked", "sum"))
        .reset_index()
    )

    # Incremental latency
    latency = (
        df[df["event_type"] == "model_inference"]
        .groupby(["experiment_id", "variant"])
        .agg(new_latency_sum=("latency_ms", "sum"))
        .reset_index()
    )

    # Merge incremental metrics
    agg = (
        new_user_counts
        .merge(clicks, on=["experiment_id", "variant"], how="outer")
        .merge(latency, on=["experiment_id", "variant"], how="outer")
        .fillna(0)
    )

    # Handle existing cumulative metrics
    metrics_path = CONFIG["paths"]["metrics_out"]
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        if "impressions" not in existing.columns:
            existing["impressions"] = 0  # backward compatibility

        cumulative = existing.groupby(["experiment_id", "variant"], as_index=False).agg(
            users=("users", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            latency_sum=("latency_sum", "sum")
        )

        agg = agg.merge(cumulative, on=["experiment_id", "variant"], how="left").fillna(0)
        agg["users"] += cumulative["users"].values
        agg["impressions"] += cumulative["impressions"].values
        agg["clicks"] += cumulative["clicks"].values
        agg["latency_sum"] += cumulative["latency_sum"].values
    else:
        agg["impressions"] = agg.get("new_impressions", 0)
        agg["users"] = agg["new_users"] + agg.get("users", 0)
        agg["clicks"] = agg["new_clicks"]
        agg["latency_sum"] = agg["new_latency_sum"]

    # Derived metrics
    agg["ctr"] = agg["clicks"] / agg["impressions"].clip(lower=1)
    agg["avg_latency_ms"] = agg["latency_sum"] / agg["impressions"].clip(lower=1)
    agg["run_id"] = datetime.now().strftime("%Y%m%d%H%M%S")

    return agg


def save_metrics(agg, df):
    """Persist cumulative metrics and update checkpoint."""
    metrics_path = CONFIG["paths"]["metrics_out"]
    cols = ["experiment_id", "variant", "users", "impressions", "clicks", "ctr",
            "latency_sum", "avg_latency_ms", "run_id"]

    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        final = pd.concat([existing, agg[cols]], ignore_index=True)
    else:
        final = agg[cols]

    final.to_csv(metrics_path, index=False)

    # Update checkpoint
    json.dump({"last_ts": df["timestamp"].max().isoformat()}, open(CONFIG["paths"]["checkpoint_ts"], "w"))

    logging.info("Incremental aggregation complete")
    logging.info(agg)

if __name__ == "__main__":
    last_ts = load_last_timestamp() # loads only events where timestamp > last_ts
    df = load_events(last_ts)

    if df.empty:
        logging.info("No new data to process")
        exit()

    seen_users = load_seen_users()
    new_users = deduplicate_users(df, seen_users)
    agg_metrics = compute_metrics(df, new_users)
    save_metrics(agg_metrics, df)