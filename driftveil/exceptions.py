class DriftError(Exception):
    """Base exception for Driftveil."""
    pass


class ContractViolationError(DriftError):
    """Exception raised when a drift contract is violated."""
    pass
