import logging
import os
import os.path as osp
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from data.dataset import build_dataset
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import networks
from configs.base import Config
from utils import optimizers, schedulers

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


class TrainEngine(object):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__()
        self.cfg = cfg
        self.mlflow_run_name = cfg.name + "-" + self.cfg.current_time
        cfg.save(
            osp.join(
                cfg.checkpoint_dir,
                "{}_{}".format(self.cfg.name, self.cfg.current_time),
                "config.json",
            )
        )

        self.logger = logging.getLogger("TrainEngine")
        self.logger.setLevel(logging.root.level)
        log_path = osp.join(cfg.checkpoint_dir, "{}_{}".format(cfg.name, cfg.current_time), "train.log")
        basedir = os.path.dirname(log_path)
        os.makedirs(basedir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.setup_mlflow(
            run_name="{}_{}".format(cfg.name, cfg.current_time),
            experiment_name="train_experiment",
        )
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            # Log configuration parameters
            mlflow.log_params(vars(cfg))

    def setup_mlflow(self, run_name: str, experiment_name: str = "Default"):
        """Set up MLflow tracking."""
        # Set experiment
        # mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
        # Start a new run
        mlflow_run = mlflow.start_run(run_name=run_name)
        self.mlflow_id = mlflow_run.info.run_id
        self.logger.info(f"MLflow run started with ID: {self.mlflow_id} and name: {self.mlflow_run_name}")
        mlflow.end_run()

    def load_train_dataset(self, car_ids):
        """Load the training dataset."""
        return build_dataset(
            self.cfg.data_root,
            brand_num=self.cfg.brand_num,
            mode="train",
            car_ids=car_ids,
            fold_num=self.cfg.fold_num,
        )

    def load_test_dataset(self, car_ids):
        """Load the training dataset."""
        return build_dataset(
            self.cfg.data_root,
            brand_num=self.cfg.brand_num,
            mode="val",
            car_ids=car_ids,
            fold_num=self.cfg.fold_num,
        )

    def get_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
    ):
        """Get dataloader for the given dataset."""

        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, list(range(os.cpu_count())))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=collate_fn,  # =collate if self.args.variable_length else None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last,
        )

    def build_model(self):
        """Build the model for training."""
        return getattr(networks, self.cfg.model_type)(self.cfg)

    def build_optimizer(self, model):
        """Build the optimizer for training."""
        return getattr(optimizers, self.cfg.optimizer)(model.parameters(), self.cfg)

    def build_scheduler(self, optimizer):
        """Build the learning rate scheduler for training."""
        return getattr(schedulers, self.cfg.lr_scheduler)(optimizer, self.cfg)

    def save_checkpoint(self, model, prefix: str = "latest"):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), prefix + ".pth")
        torch.save(model.state_dict(), ckpt_path)

    def calculate_loss(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """

        # Initialize loss functions
        if not hasattr(self, "criterion_bcel"):
            self.criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction="mean")
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="none")
        if not hasattr(self, "criterion_smoothl1"):
            self.criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction="none")

        # Classification loss
        cls_logits = predictions["logits_cls"]
        labels = targets_dict["label"]
        loss_cls = self.criterion_bcel(cls_logits, labels.float())

        inverted_labels = 1 - targets_dict["label"]
        # Mileage regression loss
        normed_mileage = targets_dict["normed_mileage"]
        logits_mileage = predictions["logits_mileage"]
        loss_mileage = self.criterion_mse(logits_mileage.squeeze(), normed_mileage)
        loss_mileage = loss_mileage * inverted_labels  # Only calculate mileage loss for normal samples (label=0)
        loss_mileage = loss_mileage.mean()

        # Reconstruction loss
        logits_rec = predictions["logits_rec"]
        normed_time_series = targets_dict["normed_time_series"]
        loss_reg = self.criterion_smoothl1(logits_rec, normed_time_series).mean(dim=[1, 2])
        loss_reg = loss_reg * inverted_labels  # Only calculate reconstruction loss for normal samples (label=0)
        loss_reg = loss_reg.mean()

        total_loss = 1.0 * loss_cls + 1.0 * (loss_mileage + loss_reg)

        return {"total_loss": total_loss, "loss_cls": loss_cls, "loss_mileage": loss_mileage, "loss_reg": loss_reg}

    def calculate_metrics(self, preds: np.ndarray, targets: np.ndarray, num_linspace: int = 1000) -> Dict:
        """Calculate metrics given predictions and targets."""
        metric_dict = {}
        for threshold in np.linspace(np.min(preds), np.max(preds), num=num_linspace):
            precision = metrics.precision_score(targets, preds >= threshold, average="binary", zero_division=0)
            recall = metrics.recall_score(targets, preds >= threshold, average="binary", zero_division=0)
            f1 = metrics.f1_score(targets, preds >= threshold, average="binary", zero_division=0)
            accuracy = metrics.accuracy_score(targets, preds >= threshold)
            if f1 > metric_dict.get("f1", -1):
                metric_dict["accuracy"] = accuracy
                metric_dict["precision"] = precision
                metric_dict["recall"] = recall
                metric_dict["f1"] = f1
                metric_dict["best_threshold"] = threshold

        return metric_dict

    def export_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray, num_linspace: int = 1000, prefix: str = ""):
        """Export confusion matrix given predictions and targets."""
        best_threshold = None
        best_f1 = -1
        for threshold in np.linspace(np.min(preds), np.max(preds), num=num_linspace):
            f1 = metrics.f1_score(targets, preds >= threshold, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        cm = metrics.confusion_matrix(targets, preds >= best_threshold)
        # Move 1s to the top-left corner and 0s to the bottom-right corner
        cm = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])
        # Plot confusion matrix
        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # Remove colorbar
        plt.gca().collections[0].colorbar.remove()
        # plt.title("Confusion Matrix")
        # Customize x and y ticks to show "Anomalous" and "Normal"
        plt.xticks([0.5, 1.5], ["Anomalous", "Normal"])
        plt.yticks([0.5, 1.5], ["Anomalous", "Normal"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        # Move xticks and xlabel to the top
        plt.gca().xaxis.set_label_position("top")
        plt.gca().xaxis.tick_top()

        # Save confusion matrix figure
        cm_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"confusion_matrix_{prefix}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

    def train_step(self, model, batch, optimizer):
        """Perform a single training step."""
        optimizer.zero_grad()
        batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
        outputs = model(batch)
        loss_dict = self.calculate_loss(outputs, batch)
        loss_dict["total_loss"].backward()
        optimizer.step()

        predictions = outputs["logits_cls"].sigmoid().squeeze().detach().cpu().numpy()
        metric_dict = self.calculate_metrics(predictions, batch["label"].detach().cpu().numpy(), num_linspace=5)

        return loss_dict, metric_dict

    def train_epoch(self, model, dataloader, optimizer, scheduler):
        model.train()
        if not hasattr(self, "global_step"):
            self.global_step = 0

        with tqdm(total=len(dataloader), ascii=True) as pbar:
            with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                for batch in dataloader:
                    self.global_step += 1
                    loss_dict, metric_dict = self.train_step(model, batch, optimizer)
                    postfix = "Epoch {}/{} - ".format(self.global_step // len(dataloader) + 1, self.cfg.num_epochs)
                    postfix += "Total Loss: {:.8f} - ".format(loss_dict["total_loss"].item())
                    postfix += "F1: {:.2f} - ".format(metric_dict["f1"])
                    pbar.set_description(postfix)
                    pbar.update(1)
                    for key, value in loss_dict.items():
                        mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
                    for key, value in metric_dict.items():
                        mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
                scheduler.step()

    def evaluate(self, model, dataset):
        """Evaluate the model on the given dataloader."""
        model.eval()

        # For each car, calculate the average score across all its samples and use that for evaluation
        car_scores_cls = {}
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
                    scores_cls = outputs["logits_cls"].sigmoid().squeeze().detach().cpu().numpy().tolist()
                    scores_rec = (
                        self.criterion_mse(outputs["logits_rec"], batch["normed_time_series"]).mean(dim=[1, 2]).detach().cpu().tolist()
                    )

                for car_id, label, score_cls, score_rec in zip(car_ids, labels, scores_cls, scores_rec):
                    if car_id not in car_scores_cls:
                        car_scores_cls[car_id] = []
                        car_scores_rec[car_id] = []
                    car_scores_cls[car_id].append(score_cls)
                    car_scores_rec[car_id].append(score_rec)
                    car_labels[car_id] = label

        # Average scores for each car
        car_avg_scores_cls = {car_id: np.mean(scores) for car_id, scores in car_scores_cls.items()}
        car_avg_scores_rec = {car_id: np.mean(scores) for car_id, scores in car_scores_rec.items()}

        # Calculate metrics based on average scores
        y_true = [car_labels[car_id] for car_id in car_avg_scores_cls.keys()]
        y_scores_cls = [car_avg_scores_cls[car_id] for car_id in car_avg_scores_cls.keys()]
        metric_dict_cls = self.calculate_metrics(np.array(y_scores_cls), np.array(y_true))

        y_scores_rec = [car_avg_scores_rec[car_id] for car_id in car_avg_scores_rec.keys()]
        metric_dict_rec = self.calculate_metrics(np.array(y_scores_rec), np.array(y_true))

        metric_dict = {}
        for key in metric_dict_cls.keys():
            metric_dict["cls_" + key] = metric_dict_cls[key]
        for key in metric_dict_rec.keys():
            metric_dict["rec_" + key] = metric_dict_rec[key]

        return metric_dict, (car_labels, car_avg_scores_cls, car_avg_scores_rec)

    def load_data(self):
        car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand3/label/all_label.csv")
        car_normal_ids = car_info[car_info["label"] == 0]["car"].unique().tolist()
        car_abnormal_ids = car_info[car_info["label"] == 1]["car"].unique().tolist()

        train_dataset = self.load_train_dataset(car_normal_ids)
        min_mileage, max_mileage = train_dataset.get_min_max_mileage()
        test_dataset = self.load_test_dataset(car_normal_ids + car_abnormal_ids)
        test_dataset.set_min_max_mileage(min_mileage, max_mileage)

        train_dataloader = self.get_dataloader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        return train_dataloader, test_dataset, car_normal_ids, car_abnormal_ids

    def run(self):
        """Run the training process."""

        train_dataloader, test_dataset, car_normal_ids, car_abnormal_ids = self.load_data()

        model = self.build_model()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        best_f1_cls = 0.0
        best_f1_rec = 0.0
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{self.cfg.num_epochs}")
            self.train_epoch(model, train_dataloader, optimizer, scheduler)
            self.save_checkpoint(model, prefix=f"latest")

            with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=self.global_step)
                metric_dict, _ = self.evaluate(model, test_dataset)
                for key, value in metric_dict.items():
                    self.logger.info(f"Test {key}: {value:.4f}")
                    mlflow.log_metric("test_{}".format(key), value, step=self.global_step)
                if metric_dict["cls_f1"] > best_f1_cls:
                    best_f1_cls = metric_dict["cls_f1"]
                    self.logger.info(f"New best F1 score for classification: {best_f1_cls:.4f}. Saving checkpoint.")
                    self.save_checkpoint(model, prefix="best_cls")
                if metric_dict["rec_f1"] > best_f1_rec:
                    best_f1_rec = metric_dict["rec_f1"]
                    self.logger.info(f"New best F1 score for reconstruction: {best_f1_rec:.4f}. Saving checkpoint.")
                    self.save_checkpoint(model, prefix="best_rec")

        self.logger.info("Training completed.")
