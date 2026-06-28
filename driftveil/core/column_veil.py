import numpy as np
import pandas as pd
from typing import List, Union

class ColumnVeil:
    """Fluent column contract builder."""
    def __init__(self, pact, column_name: str):
        self.pact = pact
        self.column_name = column_name

    def _add_contract(self, contract_type: str, params: dict, severity: str = "error"):
        self.pact._add_contract(
            target_type="column",
            target=self.column_name,
            contract_type=contract_type,
            params=params,
            severity=severity
        )
        return self

    def is_normal(self, tolerance: float = 0.10, severity: str = "error"):
        return self._add_contract("is_normal", {"tolerance": tolerance}, severity)

    def is_lognormal(self, tolerance: float = 0.10, severity: str = "error"):
        return self._add_contract("is_lognormal", {"tolerance": tolerance}, severity)

    def is_exponential(self, tolerance: float = 0.15, severity: str = "error"):
        return self._add_contract("fits_distribution", {"dist_name": "expon", "tolerance": tolerance}, severity)

    def fits_distribution(self, dist_name: str, tolerance: float = 0.10, severity: str = "error"):
        return self._add_contract("fits_distribution", {"dist_name": dist_name, "tolerance": tolerance}, severity)

    def mean_stable(self, tolerance: float = 0.15, severity: str = "error"):
        return self._add_contract("mean_stable", {"tolerance": tolerance}, severity)

    def variance_stable(self, tolerance: float = 0.20, severity: str = "error"):
        return self._add_contract("variance_stable", {"tolerance": tolerance}, severity)

    def median_stable(self, tolerance: float = 0.15, severity: str = "error"):
        return self._add_contract("median_stable", {"tolerance": tolerance}, severity)

    def quantile_stable(self, quantiles: List[float] = None, tolerance: float = 0.10, severity: str = "error"):
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75, 0.95]
        return self._add_contract("quantile_stable", {"quantiles": quantiles, "tolerance": tolerance}, severity)

    def psi_below(self, threshold: float = 0.20, severity: str = "error"):
        return self._add_contract("psi_below", {"threshold": threshold}, severity)

    def ks_pvalue_above(self, threshold: float = 0.05, severity: str = "error"):
        return self._add_contract("ks_pvalue_above", {"threshold": threshold}, severity)

    def js_divergence_below(self, threshold: float = 0.10, severity: str = "error"):
        return self._add_contract("js_divergence_below", {"threshold": threshold}, severity)

    def null_rate_stable(self, tolerance: float = 0.05, severity: str = "error"):
        return self._add_contract("null_rate_stable", {"tolerance": tolerance}, severity)

    def outlier_rate_stable(self, method: str = "iqr", tolerance: float = 0.10, severity: str = "error"):
        return self._add_contract("outlier_rate_stable", {"method": method, "tolerance": tolerance}, severity)

    def stays_in_range(self, min: float = None, max: float = None, severity: str = "error"):
        return self._add_contract("stays_in_range", {"min_val": min, "max_val": max}, severity)

    def no_new_categories(self, severity: str = "error"):
        return self._add_contract("no_new_categories", {}, severity)

    def category_freq_stable(self, chi2_pvalue: float = 0.05, severity: str = "error"):
        return self._add_contract("category_freq_stable", {"chi2_pvalue": chi2_pvalue}, severity)
