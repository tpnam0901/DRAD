import logging
import os
from abc import ABC, abstractmethod

import mlflow


class BaseEngine(ABC):
    def __init__(self, log_path: str = ""):
        """Initialize the base engine."""
        self.mlflow_id = None
        self.mlflow_run_name = None
        self.logger = logging.getLogger("Base")
        self.logger.setLevel(logging.root.level)
        if log_path:
            basedir = os.path.dirname(log_path)
            os.makedirs(basedir, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def setup_mlflow(self, run_name: str, experiment_name: str = "Default"):
        """Set up MLflow tracking."""
        # Set experiment
        # mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
        # Start a new run
        mlflow_run = mlflow.start_run(run_name=run_name)
        self.mlflow_id = mlflow_run.info.run_id
        self.mlflow_run_name = run_name
        self.logger.info(f"MLflow run started with ID: {self.mlflow_id} and name: {self.mlflow_run_name}")
        mlflow.end_run()

    @abstractmethod
    def run(self):
        """Run the engine process."""
        pass
