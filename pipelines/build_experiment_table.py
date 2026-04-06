# pipelines/build_experiment_table.py
# Reads all raw events
# Recalculates metrics from scratch
# Outputs a clean final summary table

import pandas as pd

# Load raw event logs
events = pd.read_csv("data/raw/event_logs.csv")

# Separate event types
assignments = events[events["event_type"] == "variant_assignment"]  # which variant each user got
inference = events[events["event_type"] == "model_inference"]      # model prediction events
responses = events[events["event_type"] == "user_response"]        # user click/no-click events

# Ensure users only belong to one variant
variant_check = assignments.groupby("user_id")["variant"].nunique()
assert variant_check.max() == 1, "User assigned to multiple variants!"  # safety check


# Aggregate clicks and impressions
click_table = (
    responses
    .groupby(["experiment_id", "variant"])
    .agg(
        users=("user_id", "nunique"),         # unique users per variant
        impressions=("user_id", "count"),     # total events (impressions)
        clicks=("clicked", "sum")             # total clicks
    )
    .reset_index()
)

# Compute click-through rate
click_table["ctr"] = click_table["clicks"] / click_table["impressions"].clip(lower=1)


# Aggregate latency (sum)
latency_table = (
    inference
    .groupby(["experiment_id", "variant"])
    .agg(
        latency_sum=("latency_ms", "sum")     # sum of model inference latencies
    )
    .reset_index()
)


# Merge clicks and latency
experiment_metrics = click_table.merge(
    latency_table,
    on=["experiment_id", "variant"],
    how="left"  # keep all variants from clicks table
)

# Compute average latency per impression
experiment_metrics["avg_latency_ms"] = experiment_metrics["latency_sum"] / experiment_metrics["impressions"].clip(lower=1)


# Save experiment-level metrics table
experiment_metrics.to_csv("data/processed/experiment_metrics.csv", index=False)

print("Experiment metrics table built successfully")
print(experiment_metrics)
