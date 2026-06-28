from dataclasses import dataclass
from typing import Any

@dataclass
class TestResult:
    """Result of a pure statistical test function."""
    passed: bool
    actual: Any
    expected: Any
    details: str
