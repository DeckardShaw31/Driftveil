import os
import json
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional

from driftveil.exceptions import DriftError
from driftveil.core.column_veil import ColumnVeil
from driftveil.core.pair_veil import PairVeil
from driftveil.core.dataset_veil import DatasetVeil
from driftveil.io.adapters import to_pandas
from driftveil.io.serialise import save_pact, load_pact
from driftveil.report.report import ContractReport, CheckResult

# Import test functions
from driftveil.tests import distribution, normality, stability, correlation, categorical

class DriftVeil:
    """Main Driftveil entry point representing a statistical contract."""
    
    def __init__(self, reference_df: Optional[pd.DataFrame] = None):
        self.reference_df = to_pandas(reference_df)
        self.contracts: List[Dict[str, Any]] = []

    @property
    def dataset(self) -> DatasetVeil:
        return DatasetVeil(self)

    def column(self, column_name: str) -> ColumnVeil:
        return ColumnVeil(self, column_name)

    def pair(self, col_a: str, col_b: str) -> PairVeil:
        return PairVeil(self, col_a, col_b)

    def _add_contract(
        self,
        target_type: str,
        target: Union[str, tuple, None],
        contract_type: str,
        params: dict,
        severity: str = "error"
    ):
        # Prevent duplicate identical contracts
        for contract in self.contracts:
            if (contract["target_type"] == target_type and
                contract["target"] == target and
                contract["contract_type"] == contract_type and
                contract["params"] == params):
                return
                
        self.contracts.append({
            "target_type": target_type,
            "target": target,
            "contract_type": contract_type,
            "params": params,
            "severity": severity,
            "ref_stats": None
        })

    def _compute_ref_stats(self):
        """Compute reference statistics lazily for all contracts."""
        if self.reference_df is None:
            return # Cannot compute, rely on loaded stats
            
        for contract in self.contracts:
            if contract["ref_stats"] is not None:
                continue # Already computed
                
            ttype = contract["target_type"]
            target = contract["target"]
            ctype = contract["contract_type"]
            params = contract["params"]
            
            stats = {}
            if ttype == "column":
                col = target
                if col not in self.reference_df.columns:
                    raise DriftError(f"Column '{col}' not found in reference DataFrame.")
                ref_s = self.reference_df[col]
                
                if ctype in ("is_normal", "is_lognormal"):
                    if ctype == "is_normal":
                        stats = {
                            "mean": float(ref_s.mean()) if not ref_s.isnull().all() else 0.0,
                            "std": float(ref_s.std()) if not ref_s.isnull().all() else 1.0
                        }
                    else:
                        ref_pos = ref_s[ref_s > 0]
                        stats = {
                            "log_mean": float(np.log(ref_pos).mean()) if len(ref_pos) > 0 else 0.0,
                            "log_std": float(np.log(ref_pos).std()) if len(ref_pos) > 0 else 1.0
                        }
                elif ctype == "fits_distribution":
                    dist_name = params["dist_name"]
                    import scipy.stats
                    dist = getattr(scipy.stats, dist_name, None)
                    if dist is None:
                        raise ValueError(f"Unknown scipy.stats distribution: {dist_name}")
                    clean_s = ref_s.dropna()
                    if len(clean_s) >= 3:
                        fit_params = dist.fit(clean_s)
                    else:
                        fit_params = (0.0, 1.0)
                    stats = {"fit_params": [float(x) for x in fit_params]}
                elif ctype in ("mean_stable", "variance_stable", "median_stable"):
                    stats = {
                        "mean": float(ref_s.mean()) if not ref_s.isnull().all() else 0.0,
                        "variance": float(ref_s.var()) if not ref_s.isnull().all() else 0.0,
                        "count": int(ref_s.dropna().count()),
                        "median": float(ref_s.median()) if not ref_s.isnull().all() else 0.0
                    }
                elif ctype == "quantile_stable":
                    q_list = params["quantiles"]
                    q_vals = {str(q): float(ref_s.dropna().quantile(q)) if len(ref_s.dropna()) > 0 else 0.0 for q in q_list}
                    stats = {"quantiles": q_vals}
                elif ctype in ("psi_below", "js_divergence_below"):
                    num_bins = params.get("num_bins", 10)
                    # Use get_ref_bins from distribution
                    bin_edges, ref_props = distribution.get_ref_bins(ref_s, num_bins)
                    stats = {
                        "bin_edges": [float(x) for x in bin_edges],
                        "bin_props": [float(x) for x in ref_props]
                    }
                elif ctype == "null_rate_stable":
                    stats = {"null_rate": float(ref_s.isnull().mean())}
                elif ctype == "outlier_rate_stable":
                    clean_s = ref_s.dropna()
                    if len(clean_s) >= 4:
                        q25 = float(clean_s.quantile(0.25))
                        q75 = float(clean_s.quantile(0.75))
                        iqr = q75 - q25
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        ref_outlier_rate = float(((clean_s < lower_bound) | (clean_s > upper_bound)).mean())
                    else:
                        lower_bound = -np.inf
                        upper_bound = np.inf
                        ref_outlier_rate = 0.0
                    stats = {
                        "outlier_rate": ref_outlier_rate,
                        "lower_bound": lower_bound if lower_bound != -np.inf else "-inf",
                        "upper_bound": upper_bound if upper_bound != np.inf else "inf"
                    }
                elif ctype == "stays_in_range":
                    stats = {
                        "min": float(ref_s.min()) if len(ref_s.dropna()) > 0 else -np.inf,
                        "max": float(ref_s.max()) if len(ref_s.dropna()) > 0 else np.inf
                    }
                elif ctype in ("no_new_categories", "category_freq_stable"):
                    clean_s = ref_s.dropna()
                    categories = [str(x) for x in clean_s.unique()]
                    counts = clean_s.value_counts()
                    props = (counts / counts.sum()).to_dict() if len(counts) > 0 else {}
                    props_str = {str(k): float(v) for k, v in props.items()}
                    stats = {
                        "categories": categories,
                        "category_props": props_str
                    }
                    
            elif ttype == "pair":
                col_a, col_b = target
                if col_a not in self.reference_df.columns or col_b not in self.reference_df.columns:
                    raise DriftError(f"Columns '{col_a}' or '{col_b}' not found in reference DataFrame.")
                
                if ctype == "correlation":
                    method = params["method"]
                    df_ref = self.reference_df[[col_a, col_b]].dropna()
                    corr = float(df_ref[col_a].corr(df_ref[col_b], method=method)) if len(df_ref) >= 2 else 0.0
                    stats = {
                        "corr": corr if not np.isnan(corr) else 0.0,
                        "count": len(df_ref)
                    }
                elif ctype == "mutual_information_stable":
                    df_ref = self.reference_df[[col_a, col_b]].dropna()
                    mi = 0.0
                    if len(df_ref) > 0:
                        try:
                            from sklearn.feature_selection import mutual_info_regression
                            mi = float(mutual_info_regression(df_ref[[col_a]], df_ref[col_b])[0])
                        except ImportError:
                            pass
                    stats = {"mi": mi}
                elif ctype == "ratio_stable":
                    ref_ratio_series = (self.reference_df[col_a] / self.reference_df[col_b]).replace([np.inf, -np.inf], np.nan).dropna()
                    median_ratio = float(ref_ratio_series.median()) if len(ref_ratio_series) > 0 else 0.0
                    stats = {"median_ratio": median_ratio}
                    
            elif ttype == "dataset":
                if ctype == "row_count_stable":
                    stats = {"row_count": len(self.reference_df)}
                elif ctype in ("no_new_columns", "no_dropped_columns"):
                    stats = {"columns": list(self.reference_df.columns)}
                    
            contract["ref_stats"] = stats

    def enforce(self, new_df: pd.DataFrame, raise_on_fail: bool = False) -> ContractReport:
        """Enforce all contracts on the new DataFrame."""
        new_df = to_pandas(new_df)
        self._compute_ref_stats()
        
        results: List[CheckResult] = []
        
        for contract in self.contracts:
            ttype = contract["target_type"]
            target = contract["target"]
            ctype = contract["contract_type"]
            params = contract["params"]
            severity = contract["severity"]
            ref_stats = contract["ref_stats"]
            
            # Determine target reference series/dataframe or ref_stats
            ref_source = ref_stats
            if self.reference_df is not None:
                if ttype == "column":
                    ref_source = self.reference_df[target]
                elif ttype == "pair":
                    ref_source = (self.reference_df[target[0]], self.reference_df[target[1]])
                elif ttype == "dataset":
                    ref_source = self.reference_df
                    
            # Check target columns in new_df
            missing_cols = []
            if ttype == "column":
                if target not in new_df.columns:
                    missing_cols = [target]
            elif ttype == "pair":
                if target[0] not in new_df.columns:
                    missing_cols.append(target[0])
                if target[1] not in new_df.columns:
                    missing_cols.append(target[1])
                    
            if missing_cols:
                for col in missing_cols:
                    results.append(CheckResult(
                        target=col,
                        contract=ctype,
                        passed=False,
                        expected="Column exists",
                        actual="Column missing",
                        severity=severity,
                        details=f"Column '{col}' missing from new DataFrame"
                    ))
                continue
                
            # Execute test
            try:
                if ttype == "column":
                    col = target
                    new_s = new_df[col]
                    
                    if ctype == "is_normal":
                        res = normality.is_normal(ref_source, new_s, params["tolerance"])
                    elif ctype == "is_lognormal":
                        res = normality.is_lognormal(ref_source, new_s, params["tolerance"])
                    elif ctype == "fits_distribution":
                        res = normality.fits_distribution(ref_source, new_s, params["dist_name"], params["tolerance"])
                    elif ctype == "mean_stable":
                        res = stability.mean_stable(ref_source, new_s, params["tolerance"])
                    elif ctype == "variance_stable":
                        res = stability.variance_stable(ref_source, new_s, params["tolerance"])
                    elif ctype == "median_stable":
                        res = stability.median_stable(ref_source, new_s, params["tolerance"])
                    elif ctype == "quantile_stable":
                        res = stability.quantile_stable(ref_source, new_s, params["quantiles"], params["tolerance"])
                    elif ctype == "psi_below":
                        res = distribution.psi_below(ref_source, new_s, params["threshold"])
                    elif ctype == "ks_pvalue_above":
                        res = distribution.ks_2samp(ref_source, new_s, params["threshold"])
                    elif ctype == "js_divergence_below":
                        res = distribution.js_divergence(ref_source, new_s, params["threshold"])
                    elif ctype == "null_rate_stable":
                        res = stability.null_rate_stable(ref_source, new_s, params["tolerance"])
                    elif ctype == "outlier_rate_stable":
                        res = stability.outlier_rate_stable(ref_source, new_s, params["method"], params["tolerance"])
                    elif ctype == "stays_in_range":
                        min_v = params["min_val"]
                        max_v = params["max_val"]
                        res = stability.stays_in_range(ref_source, new_s, min_v, max_v)
                    elif ctype == "no_new_categories":
                        res = categorical.no_new_categories(ref_source, new_s)
                    elif ctype == "category_freq_stable":
                        res = categorical.category_freq_stable(ref_source, new_s, params["chi2_pvalue"])
                    else:
                        raise ValueError(f"Unknown column contract type: {ctype}")
                        
                elif ttype == "pair":
                    col_a, col_b = target
                    new_s1 = new_df[col_a]
                    new_s2 = new_df[col_b]
                    
                    if ctype == "correlation":
                        res = correlation.correlation_check(
                            ref_source, new_s1, new_s2, params["threshold"], params["method"], params["direction"]
                        )
                    elif ctype == "mutual_information_stable":
                        res = correlation.mutual_information_stable(ref_source, new_s1, new_s2, params["tolerance"])
                    elif ctype == "ratio_stable":
                        res = correlation.ratio_stable(ref_source, new_df, col_a, col_b, params["ratio_name"], params["tolerance"])
                    else:
                        raise ValueError(f"Unknown pair contract type: {ctype}")
                        
                elif ttype == "dataset":
                    if ctype == "row_count_stable":
                        ref_count = ref_stats["row_count"]
                        new_count = len(new_df)
                        pct_shift = abs(new_count - ref_count) / ref_count if ref_count > 0 else 0.0
                        passed = pct_shift <= params["tolerance"]
                        details = f"row count {new_count} (within ±{params['tolerance']*100:.1f}%)"
                        res = distribution.TestResult(passed=passed, actual=new_count, expected=ref_count, details=details)
                    elif ctype == "no_new_columns":
                        ref_cols = set(ref_stats["columns"])
                        new_cols = set(new_df.columns)
                        unseen = new_cols - ref_cols
                        passed = len(unseen) == 0
                        details = f"no new columns (unseen: {list(unseen)})" if not passed else "no new columns"
                        res = distribution.TestResult(passed=passed, actual=list(unseen), expected=[], details=details)
                    elif ctype == "no_dropped_columns":
                        ref_cols = set(ref_stats["columns"])
                        new_cols = set(new_df.columns)
                        missing = ref_cols - new_cols
                        passed = len(missing) == 0
                        details = f"dropped columns found: {list(missing)}" if not passed else "no dropped columns"
                        res = distribution.TestResult(passed=passed, actual=list(missing), expected=[], details=details)
                    else:
                        raise ValueError(f"Unknown dataset contract type: {ctype}")
                        
                results.append(CheckResult(
                    target=target,
                    contract=ctype,
                    passed=res.passed,
                    expected=res.expected,
                    actual=res.actual,
                    severity=severity,
                    details=res.details
                ))
                    
            except Exception as e:
                results.append(CheckResult(
                    target=target,
                    contract=ctype,
                    passed=False,
                    expected="Execution successful",
                    actual="Execution failed",
                    severity=severity,
                    details=f"Error executing contract {ctype}: {str(e)}"
                ))
                
        report = ContractReport(results)
        if raise_on_fail:
            report.assert_passed()
        return report

    def save(self, path: str):
        """Save the pact to a JSON file."""
        self._compute_ref_stats()
        save_pact(self, path)

    @staticmethod
    def load(path: str, reference_df: Optional[pd.DataFrame] = None) -> 'DriftVeil':
        """Load a pact from a JSON file."""
        return load_pact(path, reference_df)
