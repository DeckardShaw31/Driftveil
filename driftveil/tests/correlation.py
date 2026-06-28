import numpy as np
import pandas as pd
import scipy.stats
from typing import Union, Dict, Tuple, Any
from driftveil.tests import TestResult

# Try importing sklearn, if not available make mutual info raise an import warning
try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def get_ref_corr_info(ref, method="pearson"):
    if isinstance(ref, dict):
        return ref["corr"], ref["count"]
    
    s1, s2 = ref
    df_ref = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
    count = len(df_ref)
    if count < 2:
        return 0.0, count
    corr = float(df_ref['s1'].corr(df_ref['s2'], method=method))
    if np.isnan(corr):
        corr = 0.0
    return corr, count

def correlation_check(
    ref: Union[Tuple[pd.Series, pd.Series], Dict[str, Any]],
    new_s1: pd.Series,
    new_s2: pd.Series,
    threshold: float,
    method: str = "pearson",
    direction: str = "above"
) -> TestResult:
    df_new = pd.DataFrame({'s1': new_s1, 's2': new_s2}).dropna()
    new_count = len(df_new)
    if new_count < 2:
        return TestResult(passed=False, actual=np.nan, expected=threshold, details="New series has fewer than 2 aligned non-null values")
        
    new_corr = float(df_new['s1'].corr(df_new['s2'], method=method))
    if np.isnan(new_corr):
        new_corr = 0.0
        
    ref_corr, ref_count = get_ref_corr_info(ref, method)
    
    # Direction check
    if direction == "above":
        passed = new_corr >= threshold
        dir_str = "above"
    else:
        passed = new_corr <= threshold
        dir_str = "below"
        
    # Fisher z-test to compare correlations statistically
    fisher_details = ""
    if ref_count > 3 and new_count > 3:
        # Clip correlation to avoid inf in arctanh
        r_ref = np.clip(ref_corr, -0.9999, 0.9999)
        r_new = np.clip(new_corr, -0.9999, 0.9999)
        z_ref = np.arctanh(r_ref)
        z_new = np.arctanh(r_new)
        se = np.sqrt(1 / (ref_count - 3) + 1 / (new_count - 3))
        z_stat = (z_ref - z_new) / se
        p_val = float(2 * (1 - scipy.stats.norm.cdf(abs(z_stat))))
        fisher_details = f", Fisher z-test p={p_val:.4f}"
        
    details = f"correlation={new_corr:.4f} ({dir_str} {threshold}{fisher_details})"
    return TestResult(passed=passed, actual=new_corr, expected=threshold, details=details)

def mutual_information_stable(
    ref: Union[Tuple[pd.Series, pd.Series], Dict[str, Any]],
    new_s1: pd.Series,
    new_s2: pd.Series,
    tolerance: float = 0.20
) -> TestResult:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for mutual_information_stable. Install it using 'pip install scikit-learn'.")
        
    df_new = pd.DataFrame({'s1': new_s1, 's2': new_s2}).dropna()
    if len(df_new) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New series has 0 aligned non-null values")
        
    new_mi = float(mutual_info_regression(df_new[['s1']], df_new['s2'])[0])
    
    if isinstance(ref, dict):
        ref_mi = ref["mi"]
    else:
        s1, s2 = ref
        df_ref = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
        if len(df_ref) > 0:
            ref_mi = float(mutual_info_regression(df_ref[['s1']], df_ref['s2'])[0])
        else:
            ref_mi = 0.0
            
    if ref_mi == 0:
        pct_drop = 0.0
        passed = True
    else:
        pct_drop = (ref_mi - new_mi) / ref_mi
        passed = pct_drop <= tolerance
        
    details = f"mutual info stable (ref={ref_mi:.4f} → new={new_mi:.4f}, drop={pct_drop*100:.1f}%)"
    return TestResult(passed=passed, actual=new_mi, expected=ref_mi * (1 - tolerance), details=details)

def ratio_stable(
    ref: Union[pd.DataFrame, Dict[str, Any]],
    new_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    ratio_name: str,
    tolerance: float = 0.15
) -> TestResult:
    # Computes colA/colB median ratio on reference and new.
    # New median ratio
    new_ratio_series = (new_df[col_a] / new_df[col_b]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(new_ratio_series) == 0:
        return TestResult(passed=False, actual=np.nan, expected=tolerance, details="New DataFrame ratio series is empty")
        
    new_median_ratio = float(new_ratio_series.median())
    
    if isinstance(ref, dict):
        ref_median_ratio = ref["median_ratio"]
    else:
        ref_ratio_series = (ref[col_a] / ref[col_b]).replace([np.inf, -np.inf], np.nan).dropna()
        ref_median_ratio = float(ref_ratio_series.median()) if len(ref_ratio_series) > 0 else 0.0
        
    if ref_median_ratio == 0:
        pct_shift = 0.0 if new_median_ratio == 0 else np.inf
    else:
        pct_shift = (new_median_ratio - ref_median_ratio) / ref_median_ratio
        
    abs_pct_shift = abs(pct_shift)
    passed = abs_pct_shift <= tolerance
    
    sign = "+" if pct_shift >= 0 else ""
    details = f"ratio {ratio_name} stable (ref={ref_median_ratio:.4f} → new={new_median_ratio:.4f}, {sign}{pct_shift*100:.1f}%)"
    return TestResult(passed=passed, actual=pct_shift, expected=tolerance, details=details)
