import datetime
import importlib
import json
import logging
import os
import sys
from typing import Any, List, Tuple

import numpy as np


class BaseConfig(object):
    def __init__(self):
        object.__setattr__(self, "_locked", False)

    def __setattr__(self, name: str, value: Any):
        """Override to prevent adding new attributes when locked.
        Args:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        if not hasattr(self, name) and self._locked:
            raise AttributeError(
                f"Cannot add new attribute '{name}' directly to a locked config. \n"
                f"You can only modify existing attributes. \n"
                f"If you want to add new attributes, unlock the config first."
            )
        object.__setattr__(self, name, value)

    def lock(self):
        """Lock the configuration to prevent adding new attributes but allow modifying existing ones."""
        object.__setattr__(self, "_locked", True)

    def unlock(self):
        """Unlock the configuration to allow adding new attributes and modifying existing ones."""
        object.__setattr__(self, "_locked", False)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, path: str):
        """Save configuration to a JSON file.

        Args:
            path (str): Path to the JSON file (e.g., 'config.json' or 'path/to/config.json')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config_dict = {k: v for k, v in vars(self).items()}

        with open(path, "w") as cfg_file:
            json.dump(config_dict, cfg_file, indent=4)

        logging.info(f"Configuration saved to {path}")

    def load(self, path: str):
        """Load configuration from a JSON file.
        Args:
            path (str): Path to the JSON file (e.g., 'config.json' or 'path/to/config.json')
        """

        with open(path, "r") as f:
            data_dict = json.load(f)
        lock_state = data_dict.pop("_locked", True)
        self.unlock()
        for key, value in data_dict.items():
            setattr(self, key, value)
        self._locked = lock_state

        logging.info(f"Configuration loaded from {path}")


class Config(BaseConfig):
    # Base
    def __init__(self):
        super(Config, self).__init__()
        self.name = "default"
        self.description = "default"

        # --------------------------------- Training settings
        self.seed: int = np.random.randint(0, 10000)
        self.batch_size: int = 128
        self.num_epochs: int = 100
        self.checkpoint_dir: str = "working/checkpoints/RFDBattery"
        self.current_time: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_amp: bool = False
        self.val_step_freq: int = 3000
        self.mlflow_run_id: str = None  # type: ignore

        # --------------------------------- Dataset
        self.data_root: str = "working/dataset/RFDBattery"
        self.brand_num: int = 3
        self.fold_num: int = 0
        self.num_workers: int = 8
        self.alpha: float = 2.0

        # --------------------------------- Optimizer
        self.optimizer: str = "sgd"
        self.learning_rate: float = 1e-3
        self.weight_decay: float = 1e-4
        self.betas: Tuple[float, float] = (0.9, 0.999)
        self.eps: float = 1e-8
        self.amsgard: bool = False
        self.momentum: float = 0.9

        # --------------------------------- Scheduler
        self.lr_scheduler: str = "StepLR"
        self.lr_scheduler_last_epoch: int = -1

        # StepLR
        self.lr_step_size: int = 50
        self.lr_step_gamma: float = 0.5

        # MultiStepLR
        self.lr_milestones: List[int] = [50, 100, 150, 200]
        self.lr_multistep_gamma: float = 0.1

        # ExponentialLR
        self.lr_exp_gamma: float = 0.99

        # CosineAnnealingLR
        self.lr_T_max: int = self.num_epochs
        self.lr_eta_min: float = 0.00001

        # ReduceLROnPlateau
        self.lr_plateau_mode: str = "min"
        self.lr_plateau_factor: float = 0.1
        self.lr_plateau_patience: int = 10
        self.lr_plateau_threshold: float = 0.0001
        self.lr_plateau_threshold_mode: str = "rel"
        self.lr_plateau_cooldown: int = 0
        self.lr_plateau_min_lr: float = 0
        self.lr_plateau_eps: float = 1e-08

        # CosineAnnealingWarmRestarts
        self.lr_T_0: int = 50
        self.lr_T_mult: int = 2
        self.lr_eta_min: float = 1e-6

        # PolyLR
        self.poly_exponent = 0.9

        # IdentityScheduler - No scheduler

        # --------------------------------- Models
        self.model_type: str = "None"  # Model type, e.g., DyAD, STFNet, etc.

        object.__setattr__(self, "_locked", True)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)  # type: ignore
    config = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
