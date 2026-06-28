import numpy as np
import pandas as pd
import scipy.stats
from typing import Union, Dict, List, Any
from driftveil.tests import TestResult

def get_ref_mean_var_count(ref):
    if isinstance(ref, dict):
        return ref["mean"], ref["variance"], ref["count"]
    ref_clean = ref.dropna()
    if len(ref_clean) == 0:
        return 0.0, 0.0, 0
    return float(ref_clean.mean()), float(ref_clean.var()), len(ref_clean)

def mean_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.15) -> TestResult:
    new_clean = new_series.dropna()
    new_count = len(new_clean)
    if new_count == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    ref_mean, ref_var, ref_count = get_ref_mean_var_count(ref)
    new_mean = float(new_clean.mean())
    new_var = float(new_clean.var())
    
    # Calculate pct shift from reference mean
    if ref_mean == 0:
        pct_shift = 0.0 if new_mean == 0 else np.inf
    else:
        pct_shift = (new_mean - ref_mean) / ref_mean
        
    abs_pct_shift = abs(pct_shift)
    passed = abs_pct_shift <= tolerance
    
    # Welch's t-test p-value
    if ref_count > 1 and new_count > 1 and (ref_var > 0 or new_var > 0):
        # Welch's t-test formula
        se = np.sqrt(ref_var / ref_count + new_var / new_count)
        if se > 0:
            t_stat = (ref_mean - new_mean) / se
            # degrees of freedom
            num = (ref_var / ref_count + new_var / new_count) ** 2
            den = (ref_var / ref_count) ** 2 / (ref_count - 1) + (new_var / new_count) ** 2 / (new_count - 1)
            df = num / den
            p_val = float(2 * scipy.stats.t.sf(abs(t_stat), df))
            p_details = f", Welch p={p_val:.4f}"
        else:
            p_val = 1.0
            p_details = ", Welch p=1.0000"
    else:
        p_val = 1.0
        p_details = ""
        
    sign = "+" if pct_shift >= 0 else ""
    details = f"mean stable (ref={ref_mean:.2f} → {new_mean:.2f}, {sign}{pct_shift*100:.1f}%{p_details})"
    return TestResult(passed=passed, actual=pct_shift, expected=tolerance, details=details)

def variance_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.20) -> TestResult:
    new_clean = new_series.dropna()
    new_count = len(new_clean)
    if new_count == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    ref_mean, ref_var, ref_count = get_ref_mean_var_count(ref)
    new_var = float(new_clean.var())
    
    if ref_var == 0:
        pct_shift = 0.0 if new_var == 0 else np.inf
    else:
        pct_shift = (new_var - ref_var) / ref_var
        
    abs_pct_shift = abs(pct_shift)
    passed = abs_pct_shift <= tolerance
    
    # Levene's test requires raw data
    levene_details = ""
    if not isinstance(ref, dict):
        ref_clean = ref.dropna()
        if len(ref_clean) > 1 and len(new_clean) > 1:
            try:
                stat, p_val = scipy.stats.levene(ref_clean, new_clean)
                levene_details = f", Levene p={p_val:.4f}"
            except Exception:
                pass
                
    sign = "+" if pct_shift >= 0 else ""
    details = f"variance stable (ref={ref_var:.2f} → {new_var:.2f}, {sign}{pct_shift*100:.1f}%{levene_details})"
    return TestResult(passed=passed, actual=pct_shift, expected=tolerance, details=details)

def median_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.15) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    if isinstance(ref, dict):
        ref_median = ref["median"]
    else:
        ref_median = float(ref.dropna().median())
        
    new_median = float(new_clean.median())
    
    if ref_median == 0:
        pct_shift = 0.0 if new_median == 0 else np.inf
    else:
        pct_shift = (new_median - ref_median) / ref_median
        
    abs_pct_shift = abs(pct_shift)
    passed = abs_pct_shift <= tolerance
    
    mw_details = ""
    if not isinstance(ref, dict):
        ref_clean = ref.dropna()
        if len(ref_clean) > 0 and len(new_clean) > 0:
            try:
                stat, p_val = scipy.stats.mannwhitneyu(ref_clean, new_clean)
                mw_details = f", MW U p={p_val:.4f}"
            except Exception:
                pass
                
    sign = "+" if pct_shift >= 0 else ""
    details = f"median stable (ref={ref_median:.2f} → {new_median:.2f}, {sign}{pct_shift*100:.1f}%{mw_details})"
    return TestResult(passed=passed, actual=pct_shift, expected=tolerance, details=details)

def quantile_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, quantiles: List[float], tolerance: float = 0.10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series is empty")
        
    passed = True
    quantile_details = []
    max_shift = 0.0
    
    for q in quantiles:
        if isinstance(ref, dict):
            ref_q = ref["quantiles"][str(q)]
        else:
            ref_q = float(ref.dropna().quantile(q))
            
        new_q = float(new_clean.quantile(q))
        
        if ref_q == 0:
            pct_shift = 0.0 if new_q == 0 else np.inf
        else:
            pct_shift = (new_q - ref_q) / ref_q
            
        abs_shift = abs(pct_shift)
        if abs_shift > max_shift:
            max_shift = abs_shift
        if abs_shift > tolerance:
            passed = False
            
        sign = "+" if pct_shift >= 0 else ""
        quantile_details.append(f"p{int(q*100)}:{ref_q:.2f}→{new_q:.2f}({sign}{pct_shift*100:.1f}%)")
        
    details = f"quantile stable (max shift={max_shift*100:.1f}%, details: {', '.join(quantile_details)})"
    return TestResult(passed=passed, actual=max_shift, expected=tolerance, details=details)


def null_rate_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, tolerance: float = 0.05) -> TestResult:
    new_null_rate = float(new_series.isnull().mean())
    
    if isinstance(ref, dict):
        ref_null_rate = ref["null_rate"]
    else:
        ref_null_rate = float(ref.isnull().mean())
        
    abs_diff = abs(new_null_rate - ref_null_rate)
    passed = abs_diff <= tolerance
    
    details = f"null rate stable (ref={ref_null_rate:.4f} → new={new_null_rate:.4f}, diff={abs_diff:.4f})"
    return TestResult(passed=passed, actual=abs_diff, expected=tolerance, details=details)


def outlier_rate_stable(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, method: str = "iqr", tolerance: float = 0.10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=True, actual=0.0, expected=tolerance, details="New series is empty (no outliers)")
        
    if isinstance(ref, dict):
        ref_outlier_rate = ref["outlier_rate"]
        lower_bound = ref["lower_bound"]
        upper_bound = ref["upper_bound"]
    else:
        ref_clean = ref.dropna()
        if len(ref_clean) >= 4:
            q25 = float(ref_clean.quantile(0.25))
            q75 = float(ref_clean.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            ref_outlier_rate = float(((ref_clean < lower_bound) | (ref_clean > upper_bound)).mean())
        else:
            lower_bound = -np.inf
            upper_bound = np.inf
            ref_outlier_rate = 0.0
            
    new_outliers_mask = (new_clean < lower_bound) | (new_clean > upper_bound)
    new_outlier_rate = float(new_outliers_mask.mean())
    
    abs_diff = abs(new_outlier_rate - ref_outlier_rate)
    passed = abs_diff <= tolerance
    
    details = f"outlier rate stable (ref={ref_outlier_rate:.4f} → new={new_outlier_rate:.4f}, diff={abs_diff:.4f})"
    return TestResult(passed=passed, actual=abs_diff, expected=tolerance, details=details)


def stays_in_range(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, min_val: float = None, max_val: float = None) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=True, actual=None, expected=None, details="New series is empty")
        
    if min_val is None:
        if isinstance(ref, dict):
            min_val = ref["min"]
        else:
            min_val = float(ref.dropna().min()) if len(ref.dropna()) > 0 else -np.inf
            
    if max_val is None:
        if isinstance(ref, dict):
            max_val = ref["max"]
        else:
            max_val = float(ref.dropna().max()) if len(ref.dropna()) > 0 else np.inf
            
    min_ok = (new_clean >= min_val).all()
    max_ok = (new_clean <= max_val).all()
    passed = bool(min_ok and max_ok)
    
    new_min = float(new_clean.min())
    new_max = float(new_clean.max())
    
    details = f"stays in range [{min_val}, {max_val}] (new range: [{new_min:.2f}, {new_max:.2f}])"
    return TestResult(passed=passed, actual=(new_min, new_max), expected=(min_val, max_val), details=details)

