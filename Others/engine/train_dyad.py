from typing import Dict

import mlflow
import numpy as np
import torch
from configs.DyAD import Config
from tqdm.auto import tqdm

from .train_base import TrainEngine as BaseTrainEngine


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
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="mean")
        if not hasattr(self, "criterion_smoothl1"):
            self.criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction="mean")
        if not hasattr(self, "anneal0"):
            self.anneal0 = self.cfg.dyad_anneal0

        norm_label = targets_dict["normed_mileage"]
        target = targets_dict["normed_time_series"][:, :, self.cfg.dyad_decoder_embedding_size :]

        mean_pred = predictions["mean_pred"]
        log_p = predictions["log_p"]
        mean = predictions["mean"]
        log_v = predictions["log_v"]

        label_loss = self.criterion_mse(mean_pred.squeeze(), norm_label)
        nll_loss = self.criterion_smoothl1(log_p, target)
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())

        if self.cfg.dyad_anneal_function == "logistic":
            kl_weight = self.anneal0 * float(1 / (1 + np.exp(-self.cfg.dyad_k * (self.global_step - self.cfg.dyad_x0))))
        elif self.cfg.dyad_anneal_function == "linear":
            kl_weight = self.anneal0 * min(1, self.global_step / self.cfg.dyad_x0)
        else:
            kl_weight = self.anneal0

        total_loss = (
            self.cfg.dyad_nll_weight * nll_loss
            + self.cfg.dyad_latent_label_weight * label_loss
            + kl_weight * kl_loss / targets_dict["normed_time_series"].shape[0]
        )

        return {"total_loss": total_loss, "loss_label": label_loss, "loss_nll": nll_loss, "loss_kl": kl_loss}

    def train_step(self, model, batch, optimizer):
        """Perform a single training step."""
        optimizer.zero_grad()
        batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
        outputs = model(batch)
        loss_dict = self.calculate_loss(outputs, batch)
        loss_dict["total_loss"].backward()
        optimizer.step()
        return loss_dict, {"f1": -1}

    def evaluate(self, model, dataset):
        """Evaluate the model on the given dataloader."""
        model.eval()

        # Initialize loss functions
        if not hasattr(self, "criterion_mse_eval"):
            self.criterion_mse_eval = torch.nn.MSELoss(reduction="none")

        # For each car, calculate the average score across all its samples and use that for evaluation
        car_scores_rec = {}
        car_labels = {}
        for batch in tqdm(dataset, ascii=True, desc="Evaluating"):
            if isinstance(batch, list):
                old_batch = batch
                batch = {}
                for key in old_batch[0].keys():
                    batch[key] = torch.stack([item[key] for item in old_batch], dim=0)

            car_ids = batch["car"].detach().cpu().numpy().tolist()
            labels = batch["label"].detach().cpu().numpy().tolist()
            batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(batch)
                target = batch["normed_time_series"][:, :, self.cfg.dyad_decoder_embedding_size :]
                scores_rec = self.criterion_mse_eval(outputs["log_p"], target).mean(dim=[1, 2]).detach().cpu().tolist()

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
        metric_dict_rec = self.calculate_metrics(np.array(y_true), np.array(y_scores_rec))

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

            if epoch < 10 or epoch % 10 == 0:
                with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                    mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=self.global_step)
                    metric_dict, _ = self.evaluate(model, test_dataset)
                    for key, value in metric_dict.items():
                        self.logger.info(f"Test {key}: {value:.4f}")
                        mlflow.log_metric("test_{}".format(key), value, step=self.global_step)
                    if metric_dict["rec_f1"] >= best_f1_rec:
                        best_f1_rec = metric_dict["rec_f1"]
                        self.logger.info(f"New best F1 score for reconstruction: {best_f1_rec:.4f}. Saving checkpoint.")
                        self.save_checkpoint(model, prefix="best_rec_f1")

        self.logger.info("Training completed.")
