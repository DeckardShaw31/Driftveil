import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from driftveil import DriftVeil, DriftError, ContractViolationError

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    ref_df = pd.DataFrame({
        "age": np.random.normal(30, 5, n),
        "income": np.random.lognormal(10, 0.5, n),
        "revenue": np.random.normal(5000, 1000, n),
        "category": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "clicks": np.random.randint(0, 100, n),
    })
    # Make conversions correlated with clicks
    ref_df["conversions"] = ref_df["clicks"] * 0.7 + np.random.normal(0, 5, n)
    
    # Generate new data with some drift
    new_df = pd.DataFrame({
        "age": np.random.normal(30.1, 5.1, n), # stable
        "income": np.random.lognormal(10.05, 0.49, n), # stable
        "revenue": np.random.normal(7500, 1200, n), # drifted (+50% mean)
        "category": np.random.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]), # unseen category & frequency shift
        "clicks": np.random.randint(0, 100, n),
    })
    new_df["conversions"] = np.random.normal(50, 15, n) # no correlation with clicks
    
    return ref_df, new_df

def test_quickstart_flow(sample_data):
    ref_df, new_df = sample_data
    
    pact = DriftVeil(ref_df)
    
    # Register contracts
    pact.column("age").is_normal(tolerance=0.1)
    pact.column("income").is_lognormal()
    pact.column("revenue").mean_stable(tolerance=0.15)
    pact.column("category").category_freq_stable(chi2_pvalue=0.01)
    pact.pair("clicks", "conversions").correlation_above(0.5)
    pact.dataset.row_count_stable(tolerance=0.2)
    
    report = pact.enforce(new_df, raise_on_fail=False)
    
    # Assertions on checks
    assert len(report.results) == 6
    
    # Revenue should fail mean stability
    revenue_check = [r for r in report.results if r.target == "revenue" and r.contract == "mean_stable"][0]
    assert not revenue_check.passed
    assert "mean stable" in revenue_check.details
    
    # Pair clicks/conversions correlation should fail (correlation is 0.4 now, expected >= 0.5)
    corr_check = [r for r in report.results if r.target == ("clicks", "conversions")][0]
    assert not corr_check.passed
    
    # Report properties
    assert not report.passed
    assert len(report.failed) > 0
    assert len(report.violations) == len(report.failed)

def test_structural_checks(sample_data):
    ref_df, new_df = sample_data
    pact = DriftVeil(ref_df)
    
    pact.column("age").null_rate_stable(tolerance=0.01)
    pact.column("age").outlier_rate_stable(tolerance=0.05)
    pact.column("age").stays_in_range(min=0, max=100)
    pact.column("category").no_new_categories()
    pact.dataset.no_new_columns()
    pact.dataset.no_dropped_columns()
    
    report = pact.enforce(new_df)
    
    # Category should fail no_new_categories because 'D' is unseen
    cat_check = [r for r in report.results if r.target == "category" and r.contract == "no_new_categories"][0]
    assert not cat_check.passed

def test_save_load_json_pact(sample_data):
    ref_df, new_df = sample_data
    
    pact = DriftVeil(ref_df)
    pact.column("age").is_normal(tolerance=0.1)
    pact.column("revenue").mean_stable(tolerance=0.15)
    
    # Force computation of stats by saving
    with tempfile.TemporaryDirectory() as tmpdir:
        pact_path = os.path.join(tmpdir, "pact.json")
        pact.save(pact_path)
        
        assert os.path.exists(pact_path)
        
        # Load pact without ref_df
        loaded_pact = DriftVeil.load(pact_path)
        assert len(loaded_pact.contracts) == 2
        assert loaded_pact.contracts[0]["ref_stats"] is not None
        
        # Enforce on new data should run successfully
        report = loaded_pact.enforce(new_df)
        assert len(report.results) == 2
        
        # Mean stability should still fail
        rev_check = [r for r in report.results if r.target == "revenue"][0]
        assert not rev_check.passed

def test_reporting_features(sample_data):
    ref_df, new_df = sample_data
    pact = DriftVeil(ref_df)
    pact.column("age").is_normal()
    pact.column("revenue").mean_stable()
    
    report = pact.enforce(new_df)
    
    # Test summary string
    summary_str = report.summary()
    assert "age" in summary_str
    assert "revenue" in summary_str
    
    # Test to_dict
    report_dict = report.to_dict()
    assert "passed" in report_dict
    assert "results" in report_dict
    
    # Test HTML generation
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "report.html")
        report.to_html(html_path)
        assert os.path.exists(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Driftveil" in content
            assert "revenue" in content
            
    # Test plot generation
    fig = report.plot_drifts()
    assert fig is not None
    plt.close(fig)
