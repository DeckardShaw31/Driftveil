import os
import tempfile
from typing import Any

def log_report_to_mlflow(report: Any, artifact_path: str = "drift_reports"):
    """
    Log Driftveil contract report status, numeric statistics, and HTML report to active MLflow run.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required to log reports to MLflow. Install it using 'pip install mlflow'."
        )
        
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run found. Call log_report_to_mlflow within an MLflow run.")

    # 1. Log overall summary parameters
    mlflow.log_param("driftveil_passed", int(report.passed))
    mlflow.log_param("driftveil_total_checks", len(report.results))
    mlflow.log_param("driftveil_violations_count", len(report.violations))
    
    # 2. Log individual check metrics
    for r in report.results:
        # Determine clean target name
        if r.target is None:
            target_name = "dataset"
        elif isinstance(r.target, tuple):
            target_name = f"{r.target[0]}_{r.target[1]}"
        else:
            target_name = str(r.target)
            
        metric_prefix = f"driftveil_{target_name}_{r.contract}"
        
        # Log pass/fail status as a binary metric
        mlflow.log_metric(f"{metric_prefix}_passed", 1.0 if r.passed else 0.0)
        
        # Log actual value as a metric if it is a float or integer
        if isinstance(r.actual, (int, float)) and not isinstance(r.actual, bool):
            mlflow.log_metric(f"{metric_prefix}_actual", float(r.actual))
            
    # 3. Log the interactive HTML report as an artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = os.path.join(tmpdir, "drift_report.html")
        report.to_html(report_file)
        mlflow.log_artifact(report_file, artifact_path=artifact_path)
