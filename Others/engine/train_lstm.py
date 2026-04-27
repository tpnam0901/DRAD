from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from .train_ae import TrainEngine as BaseTrainEngine


class TrainEngine(BaseTrainEngine):
    def calculate_loss(self, predictions: Dict, targets_dict: Dict) -> Dict:
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
        if not hasattr(self, "criterion_smoothl1"):
            self.criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction="none")

        inverted_labels = 1 - targets_dict["label"]
        # Mileage regression loss
        normed_mileage = targets_dict["normed_mileage"]
        logits_mileage = predictions["logits_mileage"]
        loss_mileage = self.criterion_mse(logits_mileage.squeeze(), normed_mileage)
        loss_mileage = loss_mileage * inverted_labels  # Only calculate mileage loss for normal samples (label=0)
        loss_mileage = loss_mileage.mean()

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
        loss_reg = self.criterion_smoothl1(logits_rec, normed_time_series).mean(dim=[1, 2])
        loss_reg = loss_reg * inverted_labels  # Only calculate reconstruction loss for normal samples (label=0)
        loss_reg = loss_reg.mean()

        total_loss = loss_mileage + loss_reg

        return {"total_loss": total_loss, "loss_mileage": loss_mileage, "loss_reg": loss_reg}

    def train_step(self, model, batch, optimizer):
        """Perform a single training step."""
        optimizer.zero_grad()
        batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
        outputs = model(batch)
        loss_dict = self.calculate_loss(outputs, batch)
        loss_dict["total_loss"].backward()
        optimizer.step()

        predictions = outputs["logits_rec"]
        normed_time_series = torch.stack(
            [
                batch["normed_voltage"],
                batch["normed_max_cell_voltage"],
                batch["normed_min_cell_voltage"],
            ],
            dim=-1,
        )
        scores_rec = self.criterion_mse(predictions, normed_time_series).mean(dim=[1, 2]).detach().cpu().numpy()
        metric_dict = self.calculate_metrics(scores_rec, batch["label"].detach().cpu().numpy(), num_linspace=5)

        return loss_dict, metric_dict

    def evaluate(self, model, dataset):
        """Evaluate the model on the given dataloader."""
        model.eval()

        # For each car, calculate the average score across all its samples and use that for evaluation
        car_scores_rec = {}
        car_labels = {}
        for samples in tqdm(dataset, ascii=True, desc="Evaluating"):
            for batch in samples:
                for k, v in batch.items():
                    batch[k] = v.unsqueeze(0)
                car_ids = batch["car"].detach().cpu().numpy().tolist()
                labels = batch["label"].detach().cpu().numpy().tolist()
                batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
                with torch.no_grad():
                    outputs = model(batch)
                    normed_time_series = torch.stack(
                        [
                            batch["normed_voltage"],
                            batch["normed_max_cell_voltage"],
                            batch["normed_min_cell_voltage"],
                        ],
                        dim=-1,
                    )
                    scores_rec = self.criterion_mse(outputs["logits_rec"], normed_time_series).mean(dim=[1, 2]).detach().cpu().tolist()

                for car_id, label, score_rec in zip(car_ids, labels, scores_rec):
                    if car_id not in car_scores_rec:
                        car_scores_rec[car_id] = []
                    car_scores_rec[car_id].append(score_rec)
                    car_labels[car_id] = label

        # Average scores for each car
        car_avg_scores_rec = {car_id: np.mean(scores) for car_id, scores in car_scores_rec.items()}

        # Calculate metrics based on average scores
        y_true = [car_labels[car_id] for car_id in car_avg_scores_rec.keys()]
        y_scores_rec = [car_avg_scores_rec[car_id] for car_id in car_avg_scores_rec.keys()]
        metric_dict_rec = self.calculate_metrics(np.array(y_scores_rec), np.array(y_true))

        metric_dict = {}
        for key in metric_dict_rec.keys():
            metric_dict["rec_" + key] = metric_dict_rec[key]

        return metric_dict, None
