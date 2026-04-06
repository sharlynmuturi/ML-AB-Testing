# experiments/ab_test_analysis.py

import pandas as pd
import numpy as np
import mlflow
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm


# Experiment config
EXPERIMENT_NAME = "exp_model_ab_v1"
ALPHA = 0.05             # significance level
MIN_LIFT = 0.01          # practical lift threshold (1%)


# Load aggregated experiment metrics
df = pd.read_csv("data/processed/experiment_metrics.csv")

# Get latest control and treatment variants
control = df[df["variant"] == "control"].iloc[0]
treatment = df[df["variant"] == "treatment"].iloc[0]


# Extract counts & CTRs
n_control = int(control["users"])
n_treatment = int(treatment["users"])

clicks_control = int(control["clicks"])
clicks_treatment = int(treatment["clicks"])

ctr_control = clicks_control / n_control
ctr_treatment = clicks_treatment / n_treatment


# Compute lift
absolute_lift = ctr_treatment - ctr_control
relative_lift = absolute_lift / ctr_control


# Statistical test (one-sided Z-test)
stat, p_value = proportions_ztest(
    count=[clicks_treatment, clicks_control],
    nobs=[n_treatment, n_control],
    alternative="larger"  # test if treatment CTR > control CTR
)


# 95% confidence interval for lift
se = np.sqrt(
    (ctr_control * (1 - ctr_control) / n_control) +
    (ctr_treatment * (1 - ctr_treatment) / n_treatment)
)
z = norm.ppf(1 - ALPHA)
ci_lower = absolute_lift - z * se
ci_upper = absolute_lift + z * se


# Latency guardrail
latency_control = control["latency_sum"] / n_control
latency_treatment = treatment["latency_sum"] / n_treatment

# Flag if treatment is >25% slower than control
latency_regression = latency_treatment > latency_control * 1.25


# Decision logic
ship = (
    (p_value < ALPHA) and          # statistically significant
    (absolute_lift >= MIN_LIFT) and # practically significant
    not latency_regression           # passes latency guardrail
)
decision = "SHIP" if ship else "DO NOT SHIP"


# MLflow logging
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="ab_test_analysis"):
    # Log parameters
    mlflow.log_param("alpha", ALPHA)
    mlflow.log_param("min_lift", MIN_LIFT)
    mlflow.log_param("test_type", "one_sided_z_test")

    # Log metrics
    mlflow.log_metric("ctr_control", ctr_control)
    mlflow.log_metric("ctr_treatment", ctr_treatment)
    mlflow.log_metric("absolute_lift", absolute_lift)
    mlflow.log_metric("relative_lift", relative_lift)
    mlflow.log_metric("p_value", p_value)
    mlflow.log_metric("ci_lower", ci_lower)
    mlflow.log_metric("ci_upper", ci_upper)
    mlflow.log_metric("avg_latency_control", latency_control)
    mlflow.log_metric("avg_latency_treatment", latency_treatment)
    mlflow.log_metric("users_control", n_control)
    mlflow.log_metric("users_treatment", n_treatment)

    # Log tags for reference & guardrails
    mlflow.set_tag("decision", decision)
    mlflow.set_tag("experiment_id", control["experiment_id"])
    mlflow.set_tag("guardrail_latency_regression", latency_regression)


# results
print("\n===== A/B TEST RESULTS =====\n")
print(f"Control CTR   : {ctr_control:.4f}")
print(f"Treatment CTR : {ctr_treatment:.4f}")
print(f"Absolute Lift : {absolute_lift:.4%}")
print(f"Relative Lift : {relative_lift:.2%}")
print(f"P-value       : {p_value:.6f}")
print(f"95% CI Lift   : [{ci_lower:.4%}, {ci_upper:.4%}]")
print(f"\nDecision      : {decision}")