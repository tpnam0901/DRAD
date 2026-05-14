from typing import Dict

import torch
from configs.GRU import Config

from .train_dyad import TrainEngine as BaseTrainEngine
from tqdm import tqdm
import numpy as np


class TrainEngine(BaseTrainEngine):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__(cfg)
        self.cfg = cfg

    def calculate_loss(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """

        # Initialize loss functions
        if not hasattr(self, "criterion_smoothl1"):
            self.criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction="mean")

        target = targets_dict["normed_time_series"][:, :, 2:]
        log_p = predictions["log_p"]

        nll_loss = self.criterion_smoothl1(log_p, target)

        total_loss = nll_loss

        return {"total_loss": total_loss, "loss_nll": nll_loss}

    def evaluate(self, model, dataset):
        """Evaluate the model on the given dataloader."""
        model.eval()

        # Initialize loss functions
        if not hasattr(self, "criterion_mse_eval"):
            self.criterion_mse_eval = torch.nn.MSELoss(reduction="none")

        # For each car, calculate the average score across all its samples and use that for evaluation
        car_scores_rec = {}
        car_labels = {}
        for samples in tqdm(dataset, ascii=True, desc="Evaluating"):
            for batch in samples:
                for k, v in batch.items():
                    batch[k] = v.unsqueeze(0)
                car_ids = batch["car"].detach().cpu().numpy().tolist()
                labels = batch["label"].detach().cpu().numpy().tolist()
                batch = {
                    key: value.to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    for key, value in batch.items()
                }
                with torch.no_grad():
                    outputs = model(batch)
                    target = batch["normed_time_series"][:, :, 2:]
                    scores_rec = (
                        self.criterion_mse_eval(outputs["log_p"], target)
                        .mean(dim=[1, 2])
                        .detach()
                        .cpu()
                        .tolist()
                    )

                for car_id, label, score_rec in zip(car_ids, labels, scores_rec):
                    if car_id not in car_scores_rec:
                        car_scores_rec[car_id] = []
                    car_scores_rec[car_id].append(score_rec)
                    car_labels[car_id] = label

        # Average scores for each car
        car_avg_scores_rec = {
            car_id: np.mean(scores) for car_id, scores in car_scores_rec.items()
        }

        # Calculate metrics based on average scores
        y_true = [car_labels[car_id] for car_id in car_avg_scores_rec.keys()]
        y_scores_rec = [
            car_avg_scores_rec[car_id] for car_id in car_avg_scores_rec.keys()
        ]
        metric_dict_rec = self.calculate_metrics(
            np.array(y_scores_rec), np.array(y_true)
        )

        metric_dict = {}
        for key in metric_dict_rec.keys():
            metric_dict["rec_" + key] = metric_dict_rec[key]

        return metric_dict, None
