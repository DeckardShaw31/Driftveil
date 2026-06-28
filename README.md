# Driftveil 🌌

Minimalist, zero-config data drift detection for Python. Protect your ML models and data pipelines from silent distribution shifts in 3 lines of code.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D_rC-2QvP-5X8UXfafEVp8xp2Ai90Ufl?usp=sharing)

---

## ⚡ 30-Second Quickstart

```python
import pandas as pd
from driftveil import DriftVeil

# 1. Define your statistical contract on the reference data
pact = DriftVeil(reference_df)

pact.column("age").is_normal(tolerance=0.1)
pact.column("income").is_lognormal()
pact.column("revenue").mean_stable(tolerance=0.15)
pact.column("category").category_freq_stable(chi2_pvalue=0.05)
pact.pair("clicks", "conversions").correlation_above(0.5)
pact.dataset.row_count_stable(tolerance=0.2)

# 2. Enforce the contract on your production data batch
report = pact.enforce(new_df, raise_on_fail=False)

# 3. Print the summary report
print(report.summary())
# ✓ age            normal distribution   (KS p=0.43)
# ✓ income         lognormal             (KS p=0.61)
# ✗ revenue        mean drift detected   (ref=4200 → 6800, +62%)
# ✓ category       PSI=0.08              (below 0.20 threshold)
# ✓ clicks/conv.   correlation=0.71      (above 0.50)
# ✓ row count      9821 rows             (within ±20%)
```

---

## 📊 Comparison: Why Driftveil?

Most data quality tools check only the schema (types, nulls, value ranges). Driftveil checks **statistical behavior** without the heavy configuration overhead.

| Feature | **Driftveil** | Great Expectations | Evidently | Deepchecks |
| :--- | :---: | :---: | :---: | :---: |
| **Schema / null checks** | **Yes** | Yes | Partial | Yes |
| **Distribution contracts** | **Yes** | No | Reports only | Reports only |
| **Fluent Python API** | **Yes** | YAML/JSON config | Class-heavy | Class-heavy |
| **Lightweight (≤3 deps)** | **Yes** | Very heavy | Heavy | Heavy |
| **Correlation contracts** | **Yes** | No | No | Partial |
| **Saveable pacts (JSON)** | **Yes** | Yes | No | No |

---

## 🛠️ Key Design Principles

1. **Pure functions in `tests/`**
   Every statistical test is a standalone function `(ref, new) -> TestResult`. No classes, no side effects. This makes adding custom tests extremely easy.
   
2. **Lazy fluent builder**
   Contracts accumulate lazily — nothing is computed until `.enforce()` or `.save()` is called. Defining a pact is fast even with 50+ columns.
   
3. **Statistics-only serialization**
   Saved pacts store reference statistics (mean, std, bin edges, quantiles, value counts) — **not the full reference DataFrame**. A saved pact is a tiny JSON file you can commit to git and load in CI without needing the original dataset.
   
4. **Pandas-first, Polars via adapter**
   Automatically converts any supported DataFrame type (like Polars) to a Pandas-compatible interface before running tests.

---

## 📖 Full API Capabilities

### Column Metrics
```python
# --- Distribution shape ---
pact.column("x").is_normal(tolerance=0.10)
pact.column("x").is_lognormal(tolerance=0.10)
pact.column("x").is_exponential(tolerance=0.15)
pact.column("x").fits_distribution("pareto")       # Fits any scipy.stats distribution

# --- Central tendency and spread ---
pact.column("x").mean_stable(tolerance=0.15)       # Welch t-test
pact.column("x").variance_stable(tolerance=0.20)   # Levene's test
pact.column("x").median_stable(tolerance=0.15)     # Mann-Whitney U test
pact.column("x").quantile_stable(                  # Checks quantiles within tolerance
    [0.25, 0.5, 0.75, 0.95], tolerance=0.10
)

# --- Drift detection ---
pact.column("x").psi_below(0.20)                   # Population Stability Index
pact.column("x").ks_pvalue_above(0.05)             # KS two-sample test
pact.column("x").js_divergence_below(0.10)         # Jensen-Shannon divergence

# --- Structural & Categorical ---
pact.column("x").null_rate_stable(tolerance=0.05)
pact.column("x").outlier_rate_stable(method="iqr", tolerance=0.10)
pact.column("x").stays_in_range(min=0, max=100)
pact.column("category").no_new_categories()
pact.column("category").category_freq_stable(chi2_pvalue=0.05)
```

### Pair Metrics
```python
pact.pair("price", "qty").correlation_above(0.3)
pact.pair("price", "qty").correlation_below(-0.1)
pact.pair("rev", "cost").ratio_stable("rev/cost", tolerance=0.15)
pact.pair("A", "B").mutual_information_stable(tolerance=0.20)
```

### Dataset Metrics
```python
pact.dataset.row_count_stable(tolerance=0.20)
pact.dataset.no_new_columns()
pact.dataset.no_dropped_columns()
```

### Reporting & CI/CD Integration

#### Python API
```python
report = pact.enforce(new_df, raise_on_fail=False)

# Check status programmatically
if not report.passed:
    print(f"Failed checks count: {len(report.failed)}")

# Enforce and raise error in pipelines
report.assert_passed() # raises ContractViolationError

# Save & Load pacts for CI
pact.save("production.pact.json")
pact2 = DriftVeil.load("production.pact.json", reference_df)

# Export interactive report dashboards (automatically embeds plots inline)
report.to_html("drift_report.html")

# Plot drift metrics
fig = report.plot_drifts()
fig.savefig("drift_summary.png")

# Log directly to active MLflow run
report.to_mlflow()
```

#### Command Line Interface (CLI)
Run checks directly on files (CSV, Parquet, or JSON) using saved pact files in bash or scheduler pipelines:
```bash
# Exit code is 0 on pass, or 1 on contract failure (when --raise-on-fail is set)
driftveil check new_data.csv --pact production.pact.json --raise-on-fail
```

---

## 🧪 Installation

Install Driftveil from PyPI (when published):
```bash
pip install driftveil
```
Or install it with plotting and Polars support:
```bash
pip install driftveil[all]
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
