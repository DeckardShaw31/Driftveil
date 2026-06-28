class DatasetVeil:
    """Fluent dataset contract builder."""
    def __init__(self, pact):
        self.pact = pact

    def _add_contract(self, contract_type: str, params: dict, severity: str = "error"):
        self.pact._add_contract(
            target_type="dataset",
            target=None,
            contract_type=contract_type,
            params=params,
            severity=severity
        )
        return self

    def row_count_stable(self, tolerance: float = 0.20, severity: str = "error"):
        return self._add_contract("row_count_stable", {"tolerance": tolerance}, severity)

    def no_new_columns(self, severity: str = "error"):
        return self._add_contract("no_new_columns", {}, severity)

    def no_dropped_columns(self, severity: str = "error"):
        return self._add_contract("no_dropped_columns", {}, severity)
