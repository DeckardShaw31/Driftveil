import numpy as np
import pandas as pd
from typing import Tuple

class PairVeil:
    """Fluent pair contract builder."""
    def __init__(self, pact, col_a: str, col_b: str):
        self.pact = pact
        self.col_a = col_a
        self.col_b = col_b

    def _add_contract(self, contract_type: str, params: dict, severity: str = "error"):
        self.pact._add_contract(
            target_type="pair",
            target=(self.col_a, self.col_b),
            contract_type=contract_type,
            params=params,
            severity=severity
        )
        return self

    def correlation_above(self, threshold: float, method: str = "pearson", severity: str = "error"):
        return self._add_contract("correlation", {"threshold": threshold, "method": method, "direction": "above"}, severity)

    def correlation_below(self, threshold: float, method: str = "pearson", severity: str = "error"):
        return self._add_contract("correlation", {"threshold": threshold, "method": method, "direction": "below"}, severity)

    def ratio_stable(self, ratio_name: str, tolerance: float = 0.15, severity: str = "error"):
        return self._add_contract("ratio_stable", {"ratio_name": ratio_name, "tolerance": tolerance}, severity)

    def mutual_information_stable(self, tolerance: float = 0.20, severity: str = "error"):
        return self._add_contract("mutual_information_stable", {"tolerance": tolerance}, severity)
