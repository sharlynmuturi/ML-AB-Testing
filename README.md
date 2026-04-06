# ML-Powered A/B Testing & Experimentation Pipeline

This project simulates and evaluates machine learning experiments using an **end-to-end A/B testing pipeline**.

It mimics how real-world systems (e.g., recommendation engines, ads ranking models) are tested before deployment by:

- Generating user interaction data
- Aggregating experiment metrics
- Running statistical tests
- Applying decision rules (including guardrails)
- Visualizing results in a live dashboard
- Planning future experiments using power analysis

Each run is labeled:

- `SHIP` - safe to deploy
- `DO NOT SHIP` - failed criteria
- `PENDING (UNDERPOWERED)` - insufficient data
- `PENDING (LATENCY REGRESSION)` - performance issue


## Key Concepts

1. A/B Testing

- **Control** - existing model/system  
- **Treatment** - new model/system

We measure whether the treatment improves a key metric (e.g., CTR).

2. Click-Through Rate (CTR)

CTR is the main success metric. It measures how often users interact with the system.

3. Statistical Significance

We use a **Z-test for proportions** to check if the difference between control and treatment is real or due to chance.

- **p-value < 0.05** - statistically significant
    

4. Practical Significance (Lift)

Even if statistically significant, the improvement must be meaningful:

- **Absolute lift** = difference in CTR
- **Relative lift** = % improvement over control


5. Guardrails

We ensure improvements don’t harm user experience:

- **Latency constraint** - treatment cannot be too slow
- **Power constraint** - enough users must be collected

6. Power Analysis

Before running experiments, we estimate:

> “How many users do we need to detect a meaningful improvement?”

This prevents underpowered (inconclusive) experiments.

## How to Run the Pipeline

1.  Clone the repo:
    
``` bash
git clone https://github.com/sharlynmuturi/ML-AB-Testing.git  
cd ML-AB-Testing
```
2.  Create a virtual environment:
    
``` bash
python -m venv venv  
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/Mac  
```
3.  Install dependencies:
    
``` bash
pip install -r requirements.txt
```

4.  Simulate Events

```bash
python pipelines/simulate_events.py
```
This generates synthetic user activity, assigns users to control/treatment, simulates model predictions and user clicks, outputs data/raw/event_logs.csv.

5. Incremental Aggregation

```bash
python pipelines/incremental_aggregate.py
```
Processes new data, updates users, impressions, clicks and computes CTR and latency, outputs data/processed/metrics.csv

6. Build Experiment Table (Batch)

```bash
python pipelines/build_experiment_table.py
```
Recomputes metrics from scratch. Outputs, data/processed/experiment_metrics.csv

7. A/B Test Analysis

```bash
python experiments/ab_test_analysis.py
```
Computes CTR and lift, runs Z-test, builds confidence intervals, applies guardrails and logs results to MLflow

8. Power Analysis

```bash
python experiments/power_analysis.py
```

Calculates required sample size based on baseline CTR, desired minimum improvement and target statistical power

9. Dashboard

```bash
streamlit run app.py
```

The dashboard shows CTR comparison, lift trends, latency comparison, experiment decisions and alerts for issues (underpowered, latency regression)


## MLflow Integration

All experiment runs are logged with:

- Parameters (Significance level (alpha), Minimum lift)
- Metrics (CTR (control & treatment), Lift, P-value, Latency, Sample sizes)
- Tags (Final decision, Guardrail flags)