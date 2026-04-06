# experiments/power_analysis.py

import numpy as np
from statsmodels.stats.power import NormalIndPower


# Config / inputs
alpha = 0.05            # significance level (Type I error)
power = 0.8             # desired statistical power (1 - Type II error)
ctr_control = 0.08      # baseline CTR (control group)
min_lift = 0.02         # minimum detectable absolute lift (practical significance)


# Compute effect size (Cohen's h)
ctr_treatment = ctr_control + min_lift

# Clip probabilities to valid range [0,1] before arcsin transform
ctr_control = np.clip(ctr_control, 0, 1)
ctr_treatment = np.clip(ctr_treatment, 0, 1)

# Cohen's h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
effect_size = 2 * (np.arcsin(np.sqrt(ctr_treatment)) - np.arcsin(np.sqrt(ctr_control)))


# Compute required sample size per group
analysis = NormalIndPower()
sample_size_per_group = analysis.solve_power(
    effect_size=effect_size,
    power=power,
    alpha=alpha,
    alternative='larger'  # one-sided test
)


# results
print("===== POWER ANALYSIS =====\n")
print(f"Baseline CTR       : {ctr_control:.2%}")
print(f"Minimum detectable lift : {min_lift:.2%}")
print(f"Cohen's h (effect size) : {effect_size:.4f}")
print(f"Required sample size per variant : {int(np.ceil(sample_size_per_group))} users")
