- Core insight
Great Expectations validates schema — types, nulls, value ranges. But production data fails in subtler ways: distributions shift, correlations break, ratios drift. datapact is the layer above schema: define statistical contracts on your reference data, then enforce them automatically on every new batch.

- Positioning
Feature	datapact	Great Expectations	Evidently	Deepchecks
Schema / null checks	 Yes	 Yes	 Partial	 Yes
Distribution contracts	 Yes	 No	 Reports only	 Reports only
Fluent Python API	 Yes	 YAML/JSON config	 Class-heavy	 Class-heavy
Lightweight (≤3 deps)	 Yes	 Very heavy	 Heavy	 Heavy
Correlation contracts	 Yes	 No	 No	 Partial
Saveable pacts (JSON)	 Yes	 Yes	 No	 No

- Hard dependencies
pandas ≥ 1.5
scipy ≥ 1.9
numpy ≥ 1.21

- Optional extras
polars (adapter)
matplotlib (plots)
sklearn (mutual info)
mlflow (integration)

- 30-second quickstart
from datapact import DriftVeil

pact = DriftVeil(reference_df)

pact.column("age").is_normal(tolerance=0.1)
pact.column("income").is_lognormal()
pact.column("revenue").mean_stable(tolerance=0.15)
pact.column("category").distribution_stable(psi_threshold=0.2)
pact.pair("clicks", "conversions").correlation_above(0.5)
pact.dataset.row_count_stable(tolerance=0.2)

report = pact.enforce(new_df, raise_on_fail=False)
print(report.summary())
# ✓ age            normal distribution   (KS p=0.43)
# ✓ income         lognormal             (KS p=0.61)
# ✗ revenue        mean drift detected   (ref=4200 → 6800, +62%)
# ✓ category       PSI=0.08              (below 0.20 threshold)
# ✓ clicks/conv.   correlation=0.71      (above 0.50)
# ✓ row count      9821 rows             (within ±20%)

- Full ColumnVeil API
# --- Distribution shape ---
pact.column("x").is_normal(tolerance=0.10)
pact.column("x").is_lognormal(tolerance=0.10)
pact.column("x").is_exponential(tolerance=0.15)
pact.column("x").fits_distribution("pareto")     # any scipy.stats dist

# --- Central tendency and spread ---
pact.column("x").mean_stable(tolerance=0.15)     # Welch t-test
pact.column("x").variance_stable(tolerance=0.20) # Levene's test
pact.column("x").median_stable(tolerance=0.15)   # Mann-Whitney U
pact.column("x").quantile_stable(                # check p25/p50/p75/p95
    [0.25, 0.5, 0.75, 0.95], tolerance=0.10)

# --- Drift detection ---
pact.column("x").psi_below(0.20)                 # Population Stability Index
pact.column("x").ks_pvalue_above(0.05)           # KS two-sample test
pact.column("x").js_divergence_below(0.10)       # Jensen-Shannon divergence

# --- Structural ---
pact.column("x").null_rate_stable(tolerance=0.05)
pact.column("x").outlier_rate_stable(method="iqr", tolerance=0.10)
pact.column("x").stays_in_range(min=0, max=100)

# --- Categorical columns ---
pact.column("country").no_new_categories()
pact.column("status").category_freq_stable(chi2_pvalue=0.05)

- PairVeil + DatasetVeil + report object

# --- PairVeil ---
pact.pair("price", "qty").correlation_above(0.3)
pact.pair("price", "qty").correlation_below(-0.1)
pact.pair("rev", "cost").ratio_stable("rev/cost", tolerance=0.15)
pact.pair("A", "B").mutual_information_stable(tolerance=0.20)

# --- DatasetVeil ---
pact.dataset.row_count_stable(tolerance=0.20)
pact.dataset.no_new_columns()
pact.dataset.no_dropped_columns()

# --- ContractReport ---
report = pact.enforce(new_df, raise_on_fail=False)

report.passed          # bool
report.failed          # list[ContractViolation]

for v in report.failed:
    print(v.column, v.contract, v.expected, v.actual, v.severity)

report.assert_passed() # raises ContractViolationError if any failed
report.to_dict()       # JSON-serialisable
report.to_html("drift_report.html")
report.plot_drifts()   # matplotlib figure

# --- Serialise pacts for CI pipelines ---
pact.save("production.pact.json")
pact2 = DriftVeil.load("production.pact.json", reference_df)

# --- pytest integration ---
def test_no_drift():
    pact = DriftVeil.load("pact.json", ref_df)
    assert pact.enforce(new_df).passed

- Package structure

Driftveil/
├── __init__.py          # exports DriftVeil, exceptions
│
├── core/
│   ├── pact.py          # DriftVeil — main entry point
│   ├── column_veil.py   # ColumnVeil — fluent column builder
│   ├── pair_veil.py     # PairVeil — fluent pair builder
│   └── dataset_veil.py  # DatasetVeil — row count, schema
│
├── tests/                   # pure functions: (ref, new) → TestResult
│   ├── distribution.py  # PSI, KS two-sample, JS divergence
│   ├── normality.py     # Shapiro-Wilk, D'Agostino, KS fit
│   ├── stability.py     # mean, variance, median, quantile
│   ├── correlation.py   # Pearson, Spearman, mutual info
│   └── categorical.py   # chi-square, category set checks
│
├── report/
│   ├── report.py        # ContractReport + ContractViolation
│   ├── html_report.py   # generates drift_report.html
│   └── plots.py         # matplotlib drift summary
│
├── io/
│   ├── serialise.py     # pact.save() / DriftVeil.load()
│   └── adapters.py      # pandas ↔ polars normalisation
│
└── exceptions.py        # ContractViolationError, DriftError

- Key design decisions

01/ Pure functions in tests/

Every statistical test is a standalone function (ref_series, new_series) → TestResult. No classes, no side effects. Makes individual tests trivial to unit test and makes contributions easy — anyone can add a new test file.

02/ Lazy fluent builder

Each .column() call returns a ColumnVeil that registers itself back on the parent DriftVeil. Contracts accumulate lazily — nothing is computed until .enforce() is called. This means defining a pact is fast even with 50 contracts.

03/ Statistics-only serialisation

Saved pacts store reference statistics (mean, std, bin edges, quantiles) — not the full reference DataFrame. A saved pact is a small JSON file you can commit to your repo, share across teams, and load in CI without needing the original data.

04/ Pandas-first, Polars via adapter

adapters.py converts any supported DataFrame to a pandas-compatible interface before tests run. Keeps the test functions clean and framework-agnostic, while adding Polars support without forking any logic.

05/ Severity tiers in violations

Each contract has a severity: error (hard fail), warning (soft fail), info (log only). Users can set severity per contract, e.g. .psi_below(0.2, severity="warning"). Allows gradual adoption without blocking pipelines on day one.

- PSI — the flagship algorithm

Population Stability Index is the gold standard in production ML monitoring, originating in credit risk scoring. It measures how much a distribution has shifted between reference and current data.

PSI = Σ (actual_i% − expected_i%) × ln(actual_i% / expected_i%)
     i = 1..N_bins  (default: 10 equal-width bins on reference)

Interpretation:
  PSI < 0.10  →  No significant shift      ✓
  PSI < 0.20  →  Moderate change           ⚠  investigate
  PSI ≥ 0.20  →  Significant shift         ✗  act now

Implementation note:
  Add ε = 0.0001 to each bin frequency before log() to avoid log(0).
  Use reference bin EDGES on new data (don't recompute edges).

- Distribution drift

PSI

Population Stability Index. Bins both distributions (same edges), computes weighted KL divergence. Industry standard for feature drift.
psi_below(0.2)
KS two-sample

Kolmogorov-Smirnov on two empirical CDFs. Sensitive to any shape change. Use scipy.stats.ks_2samp.
ks_pvalue_above(0.05)
Jensen-Shannon

Symmetric, bounded [0,1]. More robust than KL divergence. Best for categorical and discrete distributions.
js_divergence_below(0.1)
Normality and distribution fitting

is_normal

Fits normal to reference, KS goodness-of-fit on new. Falls back to D'Agostino-Pearson for n > 5000 (Shapiro-Wilk is slow).
is_normal(tol=0.1)
is_lognormal

Log-transforms, then runs normality check. Most common for income, prices, durations, counts.
is_lognormal()
fits_distribution

Generic: fits any scipy.stats distribution name and checks KS goodness-of-fit. Accepts "pareto", "beta", "exponential", etc.
fits_distribution("pareto")
Central tendency and spread

mean_stable

Welch two-sample t-test. Handles unequal variances. Tolerance = max allowed % shift from reference mean.
mean_stable(tol=0.15)
variance_stable

Levene's test for equality of variances — more robust than Bartlett for non-normal data. scipy.stats.levene.
variance_stable(tol=0.2)
quantile_stable

Checks p25, p50, p75, p95 all stay within tolerance. Catches tail shifts completely missed by mean tests.
quantile_stable(tol=0.1)
Pair-level tests

correlation

Pearson (or Spearman) correlation on new data must stay above/below threshold. Uses Fisher z-test to compare correlations statistically.
correlation_above(0.5)
mutual_info

Computes MI via sklearn.feature_selection on reference and new. Flags if MI drops by more than tolerance — signals relationship breakdown.
mutual_information_stable(0.2)
ratio_stable

Computes colA/colB median ratio on reference and new. Useful for business metrics like CTR, conversion rate, gross margin.
ratio_stable("A/B", tol=0.15)
Categorical

no_new_categories

Checks unique values in new data are a subset of reference. Raises immediately on any new unseen category.
no_new_categories()
category_freq

Chi-square goodness-of-fit on category frequencies. Tests if distribution over categories has shifted. scipy.stats.chisquare.
category_freq_stable()

- Build phases

0/ Repo setup

Phase 1:

Create repo, pyproject.toml(Hatch or Poetry), MIT license, README
GitHub Actions CI: pytest on Python 3.9–3.12 + coverage badge
Claim datapact on PyPI early with an empty release
Write exceptions.py and TestResult dataclass first

1/ MVP — v0.1.0

Phase 2–4

Implement distribution.py: PSI + KS two-sample (the two most-wanted)
Implement normality.py: is_normal, is_lognormal
Implement stability.py: mean, variance, quantile
Build DriftVeil + ColumnVeil fluent API with lazy evaluation
Build ContractReport with .summary(), .assert_passed()
MkDocs + Material theme docs, hosted on GitHub Pages
Publish v0.1.0 on PyPI + write launch post

2/ Expand — v0.2.0

Phase 5–8

Add PairVeil: correlation, mutual information, ratio stability
Add categorical.py: no_new_categories, chi-square freq test
Add JS divergence + generic fits_distribution
Implement pact.save() / DriftVeil.load() — JSON serialisation
HTML drift report: report.to_html() + matplotlib plots
Polars adapter in io/adapters.

3/ Ecosystem — v0.3.0+

Phase 9+

CLI: driftveil check data.csv --pact pact.json
MLflow: log violations as run metrics automatically
Airflow / Prefect operator example in docs
GitHub Action: driftveil-action for scheduled data monitoring 
dbt test adapter (driftveil_dbt plugin package)

- How to get stars

Launch framing:
"Why Great Expectations isn't enough — introducing driftveil" on Medium / Towards Data Science. That headline hits every data engineer's feed.

One killer demo:
A Jupyter notebook showing a real ML model failing silently due to feature drift that all schema checks missed — then driftveil catching it in 5 lines.

Reddit:
r/MachineLearning, r/datascience, r/Python — post the demo notebook, not just a link.

Hacker News:
"Show HN: driftveil — statistical data contracts for DataFrames". Short, PSI example in the first comment, respond to everything.

Awesome lists:
Submit to awesome-python, awesome-ml, awesome-data-quality — these drive sustained organic stars.

Integration tutorials:
"Using driftveil with dbt", "Using driftveil in Airflow" — these rank on Google and drive steady long-tail traffic for years.