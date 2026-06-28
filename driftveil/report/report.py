from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
from driftveil.exceptions import ContractViolationError

@dataclass
class ContractViolation:
    """Represents a failed contract check."""
    column: Any          # column name, pair tuple, or None for dataset
    contract: str        # contract type
    expected: Any
    actual: Any
    severity: str        # error, warning, info
    details: str

@dataclass
class CheckResult:
    """Represents the outcome of any contract check."""
    target: Any
    contract: str
    passed: bool
    expected: Any
    actual: Any
    severity: str
    details: str

class ContractReport:
    """Report summarizing the results of enforcing contracts."""
    
    def __init__(self, results: List[CheckResult]):
        self.results = results
        # Filter violations (failed checks)
        self.violations: List[ContractViolation] = [
            ContractViolation(
                column=r.target,
                contract=r.contract,
                expected=r.expected,
                actual=r.actual,
                severity=r.severity,
                details=r.details
            )
            for r in results if not r.passed
        ]

    @property
    def passed(self) -> bool:
        """Passed is True if no 'error' severity contracts failed."""
        return not any(r.severity == "error" and not r.passed for r in self.results)

    @property
    def failed(self) -> List[ContractViolation]:
        """Returns the list of all contract violations."""
        return self.violations

    def assert_passed(self):
        """Raises ContractViolationError if any 'error' severity contract failed."""
        if not self.passed:
            failed_errors = [v for v in self.violations if v.severity == "error"]
            msg = f"{len(failed_errors)} contract violation(s) detected with severity='error':\n"
            for v in failed_errors:
                msg += f"  - {v.column} ({v.contract}): {v.details}\n"
            raise ContractViolationError(msg)

    def summary(self) -> str:
        """Returns a string representation summary of the results."""
        lines = []
        for r in self.results:
            if r.passed:
                status = "✓"
            else:
                if r.severity == "error":
                    status = "✗"
                elif r.severity == "warning":
                    status = "⚠"
                else: # info
                    status = "ℹ"
            
            # Format target name
            if r.target is None:
                target_str = "dataset"
            elif isinstance(r.target, tuple):
                target_str = f"{r.target[0]}/{r.target[1]}"
            else:
                target_str = str(r.target)
                
            lines.append(f"{status} {target_str:<18} {r.contract:<22} ({r.details})")
            
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dict of the report."""
        return {
            "passed": self.passed,
            "results": [
                {
                    "target": list(r.target) if isinstance(r.target, tuple) else r.target,
                    "contract": r.contract,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in self.results
            ]
        }

    def to_html(self, path: str):
        """Generates a beautiful HTML report."""
        from driftveil.report.html_report import generate_html_report
        generate_html_report(self, path)

    def plot_drifts(self):
        """Returns a matplotlib figure summarizing the drifts."""
        from driftveil.report.plots import plot_drifts
        return plot_drifts(self)

    def to_mlflow(self, artifact_path: str = "drift_reports"):
        """Logs this report to the active MLflow run."""
        from driftveil.report.mlflow import log_report_to_mlflow
        log_report_to_mlflow(self, artifact_path)
