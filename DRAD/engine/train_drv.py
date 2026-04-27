import torch

from typing import Dict
from engine.train import TrainEngine as BaseEngine


class TrainEngine(BaseEngine):

    def calculate_loss(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """
        logits = predictions["log_p"]
        targets = targets_dict["preprocess_inputs"]
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)
        targets = targets[:, :, self.cfg.dyad_encoder_embedding_size :].float().cuda()

        nll_loss = self.loss_nll(logits, targets)
        loss = nll_loss

        return {"total_loss": loss, "nll_loss": nll_loss}

    def calculate_score(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate MSE loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated MSE loss values.
        """
        logits = predictions["log_p"]
        targets = targets_dict["preprocess_inputs"]
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)
        targets = targets[:, :, self.cfg.dyad_encoder_embedding_size :].float().cuda()
        loss = self.loss_mse(logits, targets)
        return {"score": loss.float().item()}
