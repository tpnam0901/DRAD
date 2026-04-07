import logging
import os
import os.path as osp
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import networks
from configs.DRV import Config
from data.dataset import build_dataset
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
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="none")
        if not hasattr(self, "criterion_smoothl1"):
            self.criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction="none")

        # Mileage regression loss
        normed_mileage = targets_dict["normed_mileage"]
        logits_mileage = predictions["logits_mileage"]
        loss_mileage = self.criterion_mse(logits_mileage.squeeze(), normed_mileage)
        loss_mileage = loss_mileage.mean()

        # Reconstruction loss
        logits_rec = predictions["logits_rec"]

        normed_time_series = []
        for feature in self.cfg.output_features:
            normed_time_series.append(targets_dict[feature])
        normed_time_series = torch.stack(normed_time_series, dim=2)
        loss_reg = self.criterion_smoothl1(logits_rec, normed_time_series).mean(dim=[1, 2])
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

        # predictions = outputs["logits_cls"].sigmoid().squeeze().detach().cpu().numpy()
        metric_dict = {}  # self.calculate_metrics(predictions, batch["label"].detach().cpu().numpy(), num_linspace=5)

        return loss_dict, metric_dict

    def train_epoch(self, model, dataloader, optimizer, scheduler):
        model.train()
        if not hasattr(self, "global_step"):
            self.global_step = 0
        total_loss = 0.0
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            for batch in dataloader:
                self.global_step += 1
                loss_dict, metric_dict = self.train_step(model, batch, optimizer)
                total_loss += loss_dict["total_loss"].item()
                for key, value in loss_dict.items():
                    mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
                for key, value in metric_dict.items():
                    mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
            scheduler.step()
        return total_loss / len(dataloader)

    def run(self):
        """Run the training process."""

        car_ids = []
        if self.cfg.brand_num == 3:
            meta_data = pd.read_csv(os.path.join(self.cfg.data_root, "battery_brand3", "label", "all_label.csv"))
            # Get unique car ids from meta data
            car_ids = meta_data["car"].unique().tolist()
        else:
            meta_train = pd.read_csv(os.path.join(self.cfg.data_root, f"battery_brand{self.cfg.brand_num}", "label", "train_label.csv"))
            meta_test = pd.read_csv(os.path.join(self.cfg.data_root, f"battery_brand{self.cfg.brand_num}", "label", "test_label.csv"))
            car_ids = list(set(meta_train["car"].unique().tolist() + meta_test["car"].unique().tolist()))

            # Remove car 230 from car_ids if brand_num is 2
            if self.cfg.brand_num == 2 and 230 in car_ids:
                car_ids.remove(230)
                print("Removed car 230 from training set for brand 2")
                print("Remaining car ids for training:", car_ids)

        for idx, car_id in enumerate(car_ids):
            self.logger.info(f"Loading dataset for car {car_id}")
            train_dataset = build_dataset(
                data_root=self.cfg.data_root,
                brand_num=self.cfg.brand_num,
                mode="train",
                car_id=car_id,
                logger=self.logger,
            )

            train_dataloader = self.get_dataloader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
            )

            model = self.build_model()
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer = self.build_optimizer(model)
            scheduler = self.build_scheduler(optimizer)

            with tqdm(total=self.cfg.num_epochs, ascii=True) as pbar:
                for epoch in range(1, self.cfg.num_epochs + 1):
                    total_loss = self.train_epoch(model, train_dataloader, optimizer, scheduler)
                    postfix = "Car {}/{} - Epoch {}/{} - ".format(idx + 1, len(car_ids), epoch, self.cfg.num_epochs)
                    postfix += "Total Loss: {:.8f} - ".format(total_loss)
                    pbar.set_description(postfix)
                    pbar.update(1)

                    self.save_checkpoint(model, prefix=f"{car_id}_latest")

        self.logger.info("Training completed.")
