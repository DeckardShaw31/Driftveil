import numpy as np
import pandas as pd
import scipy.stats
from typing import Union, Dict, Any
from driftveil.tests import TestResult

def get_ref_categories(ref):
    if isinstance(ref, dict):
        return set(ref["categories"])
    return set(ref.dropna().unique())

def get_ref_category_props(ref):
    if isinstance(ref, dict):
        return ref["category_props"]
    ref_clean = ref.dropna()
    if len(ref_clean) == 0:
        return {}
    counts = ref_clean.value_counts()
    props = counts / counts.sum()
    return props.to_dict()

def no_new_categories(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series) -> TestResult:
    new_clean = new_series.dropna()
    ref_cats = get_ref_categories(ref)
    new_cats = set(new_clean.unique())
    
    unseen_cats = new_cats - ref_cats
    passed = len(unseen_cats) == 0
    
    if passed:
        details = f"no new categories (categories count={len(ref_cats)})"
    else:
        # Show first 5 unseen categories for readability
        unseen_list = sorted(list(unseen_cats))
        unseen_str = ", ".join(repr(x) for x in unseen_list[:5])
        if len(unseen_list) > 5:
            unseen_str += f", and {len(unseen_list) - 5} more"
        details = f"unseen categories found: {unseen_str}"
        
    return TestResult(passed=passed, actual=len(unseen_cats), expected=0, details=details)

def category_freq_stable(
    ref: Union[pd.Series, Dict[str, Any]],
    new_series: pd.Series,
    chi2_pvalue: float = 0.05
) -> TestResult:
    new_clean = new_series.dropna()
    ref_props = get_ref_category_props(ref)
    
    if not ref_props:
        return TestResult(passed=False, actual=np.nan, expected=chi2_pvalue, details="Reference has no categories")
        
    # Filter new data to only include categories present in reference
    ref_cat_keys = list(ref_props.keys())
    new_filtered = new_clean[new_clean.isin(ref_cat_keys)]
    n_new_filtered = len(new_filtered)
    
    if n_new_filtered == 0:
        return TestResult(passed=False, actual=np.nan, expected=chi2_pvalue, details="New series has no values matching reference categories")
        
    # Build aligned observed and expected lists
    observed = []
    expected = []
    
    # Normalize ref_props just in case they don't sum to 1
    sum_props = sum(ref_props.values())
    norm_ref_props = {k: v / sum_props for k, v in ref_props.items()}
    
    for cat in ref_cat_keys:
        obs_count = int((new_filtered == cat).sum())
        exp_count = float(n_new_filtered * norm_ref_props[cat])
        observed.append(obs_count)
        expected.append(exp_count)
        
    # Run chi-square test
    try:
        # We need at least 2 categories to run a chi-square test (df >= 1)
        if len(ref_cat_keys) < 2:
            # Only one category, if new_filtered also has it then it matches perfectly
            p_val = 1.0
        else:
            stat, p_val = scipy.stats.chisquare(f_obs=observed, f_exp=expected)
            p_val = float(p_val)
    except Exception as e:
        p_val = 0.0
        
    passed = p_val >= chi2_pvalue
    details = f"category freq stable (chi2 p={p_val:.4f}, threshold={chi2_pvalue})"
    return TestResult(passed=passed, actual=p_val, expected=chi2_pvalue, details=details)
