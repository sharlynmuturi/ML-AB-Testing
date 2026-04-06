# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.stats.power import NormalIndPower
import os


st.set_page_config(page_title="A/B Test Dashboard", layout="wide")
st.title("A/B Test Dashboard")

# Power analysis helper
def compute_required_sample(ctr_control=0.08, min_lift=0.02, alpha=0.05, power=0.8):
    ctr_treatment = ctr_control + min_lift
    ctr_control = np.clip(ctr_control, 0, 1)
    ctr_treatment = np.clip(ctr_treatment, 0, 1)
    effect_size = 2 * (np.arcsin(np.sqrt(ctr_treatment)) - np.arcsin(np.sqrt(ctr_control)))
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative='larger')
    return int(np.ceil(n))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(BASE_DIR, "experiment_runs.csv")
runs_table = pd.read_csv(csv_file)


# Map CSV columns to internal names
runs_table = runs_table.rename(columns={
    "Run ID": "run_id",
    "MLflow Decision": "tags.decision",
    "Control CTR": "metrics.ctr_control",
    "Treatment CTR": "metrics.ctr_treatment",
    "Absolute Lift": "metrics.absolute_lift",
    "Relative Lift": "metrics.relative_lift",
    "P-value": "metrics.p_value",
    "Control Latency (ms)": "metrics.avg_latency_control",
    "Treatment Latency (ms)": "metrics.avg_latency_treatment",
    "Users Control": "metrics.users_control",
    "Users Treatment": "metrics.users_treatment",
    "Adjusted Decision": "adjusted_decision"
})

# Ensure numeric columns
numeric_cols = [
    "metrics.ctr_control","metrics.ctr_treatment","metrics.absolute_lift",
    "metrics.relative_lift","metrics.avg_latency_control","metrics.avg_latency_treatment",
    "metrics.users_control","metrics.users_treatment"
]
for col in numeric_cols:
    runs_table[col] = pd.to_numeric(runs_table[col], errors="coerce").fillna(0)

# Computing required sample
required_users = compute_required_sample(ctr_control=0.08, min_lift=0.02)

# Adjusting decisions based on guardrails
def adjusted_decision(row):
    n_control = row.get("metrics.users_control", 0)
    n_treatment = row.get("metrics.users_treatment", 0)
    latency_control = row.get("metrics.avg_latency_control", 0)
    latency_treatment = row.get("metrics.avg_latency_treatment", 0)

    # Power guardrail
    if n_control < required_users or n_treatment < required_users:
        return "PENDING (UNDERPOWERED)"

    # Latency guardrail
    latency_regression = latency_treatment > latency_control * 1.30 and (latency_treatment - latency_control) > 50
    if latency_regression:
        return "PENDING (LATENCY REGRESSION)"

    # Fallback to CSV logged decision if exists
    return row.get("tags.decision", "PENDING")

# Only compute if Adjusted Decision column is missing or empty
if "adjusted_decision" not in runs_table.columns or runs_table["adjusted_decision"].isna().all():
    runs_table["adjusted_decision"] = runs_table.apply(adjusted_decision, axis=1)


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


st.subheader("Decisions & Alerts")
for i, row in runs_table.iterrows():
    decision = row["adjusted_decision"]
    if decision != "SHIP":
        st.error(f"Run {row['run_id']}: Decision = {decision}")