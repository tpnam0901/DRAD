from typing import Dict

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

from .train_base import TrainEngine as BaseTrainEngine


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
        """Evaluate the model on the given dataset."""
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

    def run(self):
        """Run the training process."""
        train_dataloader, test_dataset, car_normal_ids, car_abnormal_ids = self.load_data()

        model = self.build_model()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        best_f1_rec = 0.0
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{self.cfg.num_epochs}")
            self.train_epoch(model, train_dataloader, optimizer, scheduler)
            self.save_checkpoint(model, prefix=f"latest")
            if epoch < 5 or epoch % 10 == 0:
                with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                    mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=self.global_step)
                    metric_dict, _ = self.evaluate(model, test_dataset)
                    for key, value in metric_dict.items():
                        self.logger.info(f"Test {key}: {value:.4f}")
                        mlflow.log_metric("test_{}".format(key), value, step=self.global_step)
                    if metric_dict["rec_f1"] >= best_f1_rec:
                        best_f1_rec = metric_dict["rec_f1"]
                        self.logger.info(f"New best F1 score for reconstruction: {best_f1_rec:.4f}. Saving checkpoint.")
                        self.save_checkpoint(model, prefix="best_rec")

        self.logger.info("Training completed.")
