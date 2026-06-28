import numpy as np
import pandas as pd
import scipy.stats
from typing import Union, Dict, Any
from driftveil.tests import TestResult

def get_ref_bins(ref, num_bins=10):
    if isinstance(ref, dict):
        return np.array(ref["bin_edges"]), np.array(ref["bin_props"])
    
    ref_clean = ref.dropna()
    if len(ref_clean) == 0:
        return np.array([-np.inf, np.inf]), np.array([1.0])
    
    ref_min, ref_max = ref_clean.min(), ref_clean.max()
    if ref_min == ref_max:
        bin_edges = np.array([ref_min - 0.5, ref_min + 0.5])
    else:
        bin_edges = np.linspace(ref_min, ref_max, num_bins + 1)
        
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    ref_props = ref_counts / len(ref_clean)
    return bin_edges, ref_props

def psi_below(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, threshold: float = 0.20, num_bins: int = 10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=threshold, details="New series has 0 non-null values")
        
    bin_edges, ref_props = get_ref_bins(ref, num_bins)
    
    edges_for_hist = bin_edges.copy()
    edges_for_hist[0] = -np.inf
    edges_for_hist[-1] = np.inf
    
    new_counts, _ = np.histogram(new_clean, bins=edges_for_hist)
    new_props = new_counts / len(new_clean)
    
    eps = 0.0001
    psi_val = float(np.sum((new_props - ref_props) * np.log((new_props + eps) / (ref_props + eps))))
    
    passed = psi_val < threshold
    details = f"PSI={psi_val:.4f} (below {threshold} threshold)"
    return TestResult(passed=passed, actual=psi_val, expected=threshold, details=details)

def ks_2samp(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, threshold: float = 0.05) -> TestResult:
    new_clean = new_series.dropna()
    if isinstance(ref, dict):
        raise ValueError("KS two-sample test requires the raw reference DataFrame / Series. It cannot be run from saved pact summary statistics.")
        
    ref_clean = ref.dropna()
    if len(ref_clean) == 0 or len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=threshold, details="Empty reference or new series")
        
    stat, p_val = scipy.stats.ks_2samp(ref_clean, new_clean)
    passed = p_val >= threshold
    details = f"KS p={p_val:.4f} (threshold={threshold})"
    return TestResult(passed=passed, actual=p_val, expected=threshold, details=details)

def js_divergence(ref: Union[pd.Series, Dict[str, Any]], new_series: pd.Series, threshold: float = 0.10, num_bins: int = 10) -> TestResult:
    new_clean = new_series.dropna()
    if len(new_clean) == 0:
        return TestResult(passed=False, actual=np.nan, expected=threshold, details="New series has 0 non-null values")
        
    bin_edges, ref_props = get_ref_bins(ref, num_bins)
    
    edges_for_hist = bin_edges.copy()
    edges_for_hist[0] = -np.inf
    edges_for_hist[-1] = np.inf
    
    new_counts, _ = np.histogram(new_clean, bins=edges_for_hist)
    new_props = new_counts / len(new_clean)
    
    import scipy.spatial.distance
    js_distance = scipy.spatial.distance.jensenshannon(ref_props, new_props, base=2.0)
    js_div = float(js_distance ** 2)
    
    passed = js_div < threshold
    details = f"JS divergence={js_div:.4f} (below {threshold} threshold)"
    return TestResult(passed=passed, actual=js_div, expected=threshold, details=details)
