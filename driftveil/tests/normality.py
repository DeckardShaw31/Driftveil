import numpy as np
import pandas as pd
import scipy.stats
from typing import Union, Dict, Any
from driftveil.tests import TestResult

def get_ref_norm_params(ref):
    if isinstance(ref, dict):
        return ref["mean"], ref["std"]
    ref_clean = ref.dropna()
    if len(ref_clean) == 0:
        return 0.0, 1.0
    return float(ref_clean.mean()), float(ref_clean.std())

def is_normal(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    ref_mean, ref_std = get_ref_norm_params(ref)
    
    # 1. KS goodness-of-fit against fitted normal distribution
    if ref_std <= 0:
        # Constant reference data
        stat, ks_p = (0.0, 1.0) if (new_clean == ref_mean).all() else (1.0, 0.0)
    else:
        stat, ks_p = scipy.stats.kstest(new_clean, 'norm', args=(ref_mean, ref_std))
        
    # 2. Shapiro-Wilk or D'Agostino-Pearson normality test on the new series
    if len(new_clean) >= 3:
        if len(new_clean) <= 5000:
            _, norm_p = scipy.stats.shapiro(new_clean)
            norm_test = "SW"
        else:
            # normaltest can fail or warn on constant / too small data
            try:
                _, norm_p = scipy.stats.normaltest(new_clean)
                norm_test = "DP"
            except Exception:
                norm_p = 0.0
                norm_test = "DP"
    else:
        norm_p = 1.0
        norm_test = "SW"
        
    # We pass if the KS goodness-of-fit p-value is above the tolerance
    passed = ks_p >= tolerance
    details = f"normal distribution (KS p={ks_p:.2f}, {norm_test} p={norm_p:.2f})"
    return TestResult(passed=passed, actual=ks_p, expected=tolerance, details=details)

def is_lognormal(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.10) -> TestResult:
    # Lognormal check: log-transform and run normality check
    # Check if there are non-positive values. If so, filter or shift them.
    # To be safe, we only take values > 0.
    if isinstance(ref, dict):
        # If reference is a dict, it will have the pre-computed mean/std of log(ref)
        ref_log_stats = {
            "mean": ref["log_mean"],
            "std": ref["log_std"]
        }
    else:
        ref_clean = ref.dropna()
        ref_pos = ref_clean[ref_clean > 0]
        if len(ref_pos) == 0:
            ref_log = pd.Series([0.0])
        else:
            ref_log = np.log(ref_pos)
        ref_log_stats = ref_log
        
    new_clean = new_series.dropna()
    new_pos = new_clean[new_clean > 0]
    if len(new_pos) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series has no positive values for log-transform")
        
    new_log = np.log(new_pos)
    
    # Run normality on the log-transformed data
    res = is_normal(ref_log_stats, new_log, tolerance=tolerance)
    details = f"lognormal (KS p={res.actual:.2f})"
    return TestResult(passed=res.passed, actual=res.actual, expected=tolerance, details=details)

def fits_distribution(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, dist_name: str, tolerance: float = 0.10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    dist = getattr(scipy.stats, dist_name, None)
    if dist is None:
        raise ValueError(f"Unknown scipy.stats distribution: {dist_name}")
        
    if isinstance(ref, dict):
        fit_params = ref["fit_params"]
    else:
        ref_clean = ref.dropna()
        if len(ref_clean) < 3:
            fit_params = (0.0, 1.0)
        else:
            fit_params = dist.fit(ref_clean)
            
    # KS test of new series against the fitted distribution
    stat, ks_p = scipy.stats.kstest(new_clean, dist_name, args=fit_params)
    passed = ks_p >= tolerance
    details = f"fits {dist_name} (KS p={ks_p:.4f})"
    return TestResult(passed=passed, actual=ks_p, expected=tolerance, details=details)
