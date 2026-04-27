import os.path as osp
from typing import Dict, List, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from tqdm.auto import tqdm

import networks
from configs.base import Config
from data.naobop_dataset import TrainNaoBopDataset
from engine.base import BaseEngine
from utils.dataloader import get_dataloader
from utils.schedulers import CosineAnnealingLR


class TrainEngine(BaseEngine):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__(osp.join(cfg.ckpt_dir, "{}_{}".format(cfg.name, cfg.current_time), "train.log"))
        self.cfg = cfg
        cfg.save(osp.join(cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "config.json"))
        self.setup_mlflow(run_name="{}_{}".format(cfg.name, cfg.current_time), experiment_name="train_experiment")
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            # Log configuration parameters
            mlflow.log_params(vars(cfg))

        self.loss_nll = torch.nn.SmoothL1Loss(reduction="mean")
        self.loss_mse = torch.nn.MSELoss(reduction="mean")
        self.best_val_loss = float("inf")
        self.step = 1

    def build_model(self):
        """Build the model for training."""
        # self.logger.info("Building the model.")
        # Model building logic would go here
        model_class = getattr(networks, self.cfg.model_type)
        return model_class(self.cfg)

    def load_dataset(
        self,
        data_root: str,
        fold_num: int,
        max_length: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        car_id: int = 0,
    ) -> Dict:
        """Build the training dataset.

        Args:
            batch_size (int): Batch size for the dataloaders.
            shuffle (bool): Whether to shuffle the training data.
            num_workers (int): Number of worker threads for data loading.
            pin_memory (bool): Whether to use pinned memory for data loading.
        Returns:
            Dict: A dictionary containing training and validation datasets and dataloaders.
        """
        self.logger.info("Building training dataset.")
        train_dataset = TrainNaoBopDataset(
            data_root,
            f"fold_{fold_num}_train.txt",
            max_length,
            car_id=car_id,
        )
        self.logger.info("Building validation dataset.")
        return {
            "train_dataset": train_dataset,
            "train_loader": get_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        }

    def calculate_loss(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """
        logits = predictions["log_p"]
        mean = predictions["mean"]
        log_v = predictions["log_v"]
        targets = targets_dict["preprocess_inputs"]
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)
        targets = targets[:, :, self.cfg.dyad_decoder_embedding_size :].float().cuda()

        nll_loss = self.loss_nll(logits, targets)
        kl_loss = torch.tensor(0.0).float().cuda()
        if log_v is not None and mean is not None:
            kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
        anneal0 = 0.1
        x0 = 500
        kl_weight = anneal0 * min(1, self.step / x0)

        # This version does not use mileage labels for latent loss
        nll_weight = 10.0
        loss = nll_weight * nll_loss + kl_weight * kl_loss / logits.shape[0]

        return {"total_loss": loss, "nll_loss": nll_loss, "kl_loss": kl_loss}

    def forward(self, model, batch):
        """Perform a forward pass through the model.

        Args:
            model: The model to perform the forward pass on.
            batch: The input batch data.
        Returns:
            The model's output predictions.
        """
        return model(batch)

    def save_checkpoint(self, model, epoch: int, keep_only_latest: bool = True, prefix: str = "") -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time))
        if keep_only_latest:
            ckpt_file = osp.join(ckpt_path, prefix + "latest.pth")
        else:
            ckpt_file = osp.join(ckpt_path, prefix + f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_file)

    def train_epoch(self, model, dataloader, optimizer, scheduler, prefix="") -> float:
        loss_epoch = 0.0
        loss_dict = {}
        model.train()
        model.cuda()
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            for batch in dataloader:
                optimizer.zero_grad()
                # Forward pass
                outputs = self.forward(model, batch)
                # Compute loss
                loss_all = self.calculate_loss(outputs, batch)
                # Backward pass and optimization

                loss_all["total_loss"].backward()
                optimizer.step()
                self.step += 1

                loss_epoch += loss_all["total_loss"].item()

                desc = f"Loss: {loss_all['total_loss'].item():.4f}"
                for key, value in loss_all.items():
                    if key != "total_loss":
                        desc += f", {key}: {value.item():.4f}"
                        loss_dict[key] = loss_dict.get(key, 0.0) + value.item()

            scheduler.step()
            mlflow.log_metric(f"{prefix}learning_rate", scheduler.get_last_lr()[0], step=self.step)
            mlflow.log_metric(f"{prefix}train_epoch_loss", loss_epoch / len(dataloader), step=self.step)
            for key, value in loss_dict.items():
                mlflow.log_metric(f"{prefix}train_{key}_epoch", value / len(dataloader), step=self.step)
        return loss_epoch / len(dataloader)

    def run(self):
        """Run the training process."""

        car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand3/label/all_label.csv")
        # car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand2/fold_label.csv")
        car_ids = car_info["car"].unique().tolist()

        for idx, car_id in enumerate(car_ids):
            datasets = self.load_dataset(
                self.cfg.data_root,
                self.cfg.fold_num,
                self.cfg.max_length,
                self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                car_id=car_id,
            )
            train_loader = datasets["train_loader"]
            self.logger.info(f"Starting training for car {car_id} ({idx + 1}/{len(car_ids)})")
            model = self.build_model()
            optimizer = SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.99, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, self.cfg)

            with tqdm(total=self.cfg.num_epochs, unit="epoch") as pbar:
                for epoch in range(self.cfg.num_epochs):
                    loss_epoch = self.train_epoch(model, train_loader, optimizer, scheduler, prefix=f"car_{car_id}_")
                    self.save_checkpoint(model, epoch + 1, keep_only_latest=True, prefix=f"car_{car_id}_")
                    pbar.set_description(f"Car {car_id} Training - Loss: {loss_epoch:.4f}")
                    pbar.update(1)
        self.logger.info("Training completed.")
