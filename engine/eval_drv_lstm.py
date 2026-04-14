import os.path as osp
from typing import Dict

import torch

import networks

from .eval_drv import EvaluateEngine as EvaluateEngineBase


class EvaluateEngine(EvaluateEngineBase):
    def build_model(self):
        """Build the model for training."""
        return networks.LSTM(self.cfg)

    def calculate_score(self, predictions: Dict, targets_dict: Dict):
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """

        # Initialize loss functions
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="none")

        # Reconstruction loss
        logits_rec = predictions["logits_rec"]

        normed_time_series = torch.stack(
            [
                targets_dict["normed_voltage"],
                targets_dict["normed_max_cell_voltage"],
                targets_dict["normed_min_cell_voltage"],
            ],
            dim=-1,
        )
        loss_reg = self.criterion_mse(logits_rec, normed_time_series).mean(dim=[1, 2])

        return loss_reg

    def load_checkpoint(self, model, prefix: str = "latest"):
        """Load model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "best_rec.pth")
        model.load_state_dict(torch.load(ckpt_path))
