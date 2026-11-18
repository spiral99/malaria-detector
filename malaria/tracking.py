"""Thin MLflow wrapper. Tracking must never break a run, so every call is best-effort."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from malaria.config import ROOT


@contextmanager
def run(name: str, params: dict | None = None, experiment: str = "malaria-cross-site"):
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix()))
        mlflow.set_experiment(experiment)
        active = mlflow.start_run(run_name=name)
        if params:
            mlflow.log_params({k: str(v)[:240] for k, v in params.items()})
    except Exception as exc:
        print(f"[tracking] MLflow unavailable ({type(exc).__name__}: {exc}); continuing without it")
        mlflow, active = None, None
    tracker = Tracker(mlflow if active else None)
    try:
        yield tracker
    finally:
        if active:
            try:
                mlflow.end_run()
            except Exception:
                pass


class Tracker:
    def __init__(self, mlflow):
        self.mlflow = mlflow

    def metrics(self, values: dict, prefix: str = "") -> None:
        if not self.mlflow:
            return
        for key, value in values.items():
            try:
                if isinstance(value, (int, float)) and value == value:
                    self.mlflow.log_metric(f"{prefix}{key}", float(value))
            except Exception:
                pass

    def artifact(self, path: str | Path) -> None:
        if self.mlflow and Path(path).exists():
            try:
                self.mlflow.log_artifact(str(path))
            except Exception:
                pass
