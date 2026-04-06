# app.py

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from statsmodels.stats.power import NormalIndPower


# Page setup
st.set_page_config(page_title="A/B Test Dashboard", layout="wide")
st.title("A/B Test Dashboard")


# Auto-refresh
refresh_interval = st.sidebar.slider("Auto Refresh Interval (seconds)", 0, 60, 10)
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh")


# Power analysis helper
# Computes required sample size per variant
def compute_required_sample(ctr_control=0.08, min_lift=0.02, alpha=0.05, power=0.8):
    ctr_treatment = ctr_control + min_lift
    ctr_control = np.clip(ctr_control, 0, 1)
    ctr_treatment = np.clip(ctr_treatment, 0, 1)

    # Cohen's h effect size
    effect_size = 2 * (np.arcsin(np.sqrt(ctr_treatment)) - np.arcsin(np.sqrt(ctr_control)))
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative='larger')
    return int(np.ceil(n))


# Select experiment from MLflow
experiments = mlflow.search_experiments()
if not experiments:
    st.warning("No experiments found in MLflow.")
    st.stop()

experiment_names = [exp.name for exp in experiments]
experiment_ids = [exp.experiment_id for exp in experiments]

selected_idx = st.selectbox(
    "Select Experiment",
    range(len(experiment_names)),
    format_func=lambda i: experiment_names[i]
)
exp_id = experiment_ids[selected_idx]


# Fetch MLflow runs for experiment
runs = mlflow.search_runs(experiment_ids=[exp_id])
if runs.empty:
    st.warning("No runs found for this experiment.")
    st.stop()

# Select only the columns needed for dashboard
runs_table = runs[[
    "run_id",
    "tags.decision",
    "metrics.ctr_control",
    "metrics.ctr_treatment",
    "metrics.absolute_lift",
    "metrics.relative_lift",
    "metrics.p_value",
    "metrics.avg_latency_control",
    "metrics.avg_latency_treatment",
    "metrics.users_control",
    "metrics.users_treatment"
]].copy()


# Ensure numeric values
numeric_cols = [
    "metrics.ctr_control","metrics.ctr_treatment","metrics.absolute_lift",
    "metrics.relative_lift","metrics.avg_latency_control","metrics.avg_latency_treatment",
    "metrics.users_control","metrics.users_treatment"
]
for col in numeric_cols:
    runs_table[col] = pd.to_numeric(runs_table[col], errors="coerce").fillna(0)


# Compute minimum sample size for power guardrail
required_users = compute_required_sample(ctr_control=0.08, min_lift=0.02)


# Adjust decision based on power and latency guardrails
def adjusted_decision(row):
    n_control = row.get("metrics.users_control", 0)
    n_treatment = row.get("metrics.users_treatment", 0)

    # Power guardrail
    if n_control < required_users or n_treatment < required_users:
        return "PENDING (UNDERPOWERED)"

    # Latency guardrail (treatment cannot slow >30%)
    latency_regression = row["metrics.avg_latency_treatment"] > row["metrics.avg_latency_control"] * 1.30 and (latency_treatment - latency_control) > 50
    if latency_regression:
        return "PENDING (LATENCY REGRESSION)"

    # Fallback to MLflow logged decision
    return row.get("tags.decision", "PENDING")

runs_table["adjusted_decision"] = runs_table.apply(adjusted_decision, axis=1)


# KPI summary (latest valid run)
st.subheader("Experiment Summary")
valid_runs = runs_table[runs_table["adjusted_decision"] != "PENDING (UNDERPOWERED)"]
latest_run = valid_runs.iloc[-1] if not valid_runs.empty else runs_table.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control CTR", f"{latest_run['metrics.ctr_control']*100:.2f}%")
col2.metric("Treatment CTR", f"{latest_run['metrics.ctr_treatment']*100:.2f}%")
col3.metric("Absolute Lift", f"{latest_run['metrics.absolute_lift']*100:.2f}%",
            delta=f"{latest_run['metrics.relative_lift']*100:.2f}%")
col4.metric("Decision", latest_run["adjusted_decision"],
            delta_color="inverse" if latest_run["adjusted_decision"] != "SHIP" else "normal")


# Detailed runs table
st.subheader("Detailed Experiment Runs")
st.dataframe(runs_table.rename(columns={
    "run_id": "Run ID",
    "tags.decision": "MLflow Decision",
    "adjusted_decision": "Adjusted Decision",
    "metrics.ctr_control": "Control CTR",
    "metrics.ctr_treatment": "Treatment CTR",
    "metrics.absolute_lift": "Absolute Lift",
    "metrics.relative_lift": "Relative Lift",
    "metrics.p_value": "P-value",
    "metrics.avg_latency_control": "Control Latency (ms)",
    "metrics.avg_latency_treatment": "Treatment Latency (ms)",
    "metrics.users_control": "Users Control",
    "metrics.users_treatment": "Users Treatment"
}))


# CTR comparison chart
st.subheader("CTR Comparison by Variant")
ctr_fig = px.bar(
    runs_table.melt(
        id_vars=["run_id"],
        value_vars=["metrics.ctr_control", "metrics.ctr_treatment"],
        var_name="Variant",
        value_name="CTR"
    ),
    x="run_id",
    y="CTR",
    color="Variant",
    barmode="group",
    title="Control vs Treatment CTR",
    text_auto=".2%"
)
st.plotly_chart(ctr_fig, use_container_width=True)


# Absolute lift over time
st.subheader("Absolute Lift per Run")
lift_fig = px.line(
    runs_table,
    x="run_id",
    y="metrics.absolute_lift",
    title="Absolute Lift Over Time",
    markers=True,
    text=runs_table["metrics.absolute_lift"].apply(lambda x: f"{x*100:.2f}%")
)
st.plotly_chart(lift_fig, use_container_width=True)


# Model latency comparison
st.subheader("Model Latency")
lat_fig = px.bar(
    runs_table.melt(
        id_vars=["run_id"],
        value_vars=["metrics.avg_latency_control", "metrics.avg_latency_treatment"],
        var_name="Variant",
        value_name="Latency (ms)"
    ),
    x="run_id",
    y="Latency (ms)",
    color="Variant",
    barmode="group",
    title="Average Latency Comparison"
)
st.plotly_chart(lat_fig, use_container_width=True)


# Decisions & alerts table
st.subheader("Decisions & Alerts")
for i, row in runs_table.iterrows():
    decision = row["adjusted_decision"]
    if decision != "SHIP":
        st.error(f"Run {row['run_id']}: Decision = {decision}")

