"""
Automated experiment tracking with MLflow and Neptune.
"""

import mlflow
import neptune

class MLflowTracker:
    def start_run(self, params: dict):
        mlflow.start_run()
        mlflow.log_params(params)
    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
    def end_run(self):
        mlflow.end_run()

class NeptuneTracker:
    def __init__(self, project: str):
        self.run = neptune.init_run(project=project)
    def log_params(self, params: dict):
        self.run['parameters'] = params
    def log_metrics(self, metrics: dict):
        self.run['metrics'] = metrics
    def stop(self):
        self.run.stop()
