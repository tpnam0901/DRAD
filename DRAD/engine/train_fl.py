import os.path as osp
from typing import Dict

import pandas as pd
import torch
from torch.optim import SGD
from tqdm.auto import tqdm

from data.naobop_dataset import TrainNaoBopDataset
from engine.train_drv import TrainEngine as BaseEngine
from utils.dataloader import get_dataloader
from utils.schedulers import CosineAnnealingLR


class TrainEngine(BaseEngine):
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
        # self.logger.info("Building training dataset.")
        train_dataset = TrainNaoBopDataset(
            data_root,
            f"fold_{fold_num}_train.txt",
            max_length,
            car_id=car_id,
        )
        # self.logger.info("Building validation dataset.")
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

    def save_checkpoint(self, model, epoch: int, keep_only_latest: bool = True, prefix: str = "") -> str:
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time))
        if keep_only_latest:
            ckpt_file = osp.join(ckpt_path, prefix + "latest.pth")
        else:
            ckpt_file = osp.join(ckpt_path, prefix + f"_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_file)
        return ckpt_file

    def run(self):
        """Run the training process."""

        car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand3/label/all_label.csv")
        car_normal_ids = car_info[car_info["label"] == 0]["car"].unique().tolist()

        model = self.build_model()
        global_ckpt = self.save_checkpoint(model, 0, keep_only_latest=False, prefix="global_")
        optimizer = SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.99, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, self.cfg)

        car_dict = {car_id: ("", 0) for car_id in car_normal_ids}
        for round in range(1000):
            total_loss = 0.0
            with tqdm(total=len(car_normal_ids), unit="Car") as pbar:
                for idx, car_id in enumerate(car_normal_ids):
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
                    # self.logger.info(f"Starting training for car {car_id} ({idx + 1}/{len(car_normal_ids)})")
                    model = self.build_model()
                    model.load_state_dict(torch.load(global_ckpt))
                    optimizer = SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.99, weight_decay=1e-4)
                    ckpt_path = self.save_checkpoint(model, 0, keep_only_latest=True, prefix=f"car_{car_id}_")
                    local_loss = 0.0
                    for epoch in range(5):
                        loss_epoch = self.train_epoch(model, train_loader, optimizer, scheduler, prefix=f"car_{car_id}_")
                        local_loss += loss_epoch
                        ckpt_path = self.save_checkpoint(model, epoch + 1, keep_only_latest=True, prefix=f"car_{car_id}_")
                    local_loss /= 5
                    total_loss += local_loss
                    pbar.set_description(f"Car {car_id} - Loss: {local_loss:.4f}")
                    pbar.update(1)
                    car_dict[car_id] = (ckpt_path, len(datasets["train_dataset"]))
            total_loss /= len(car_normal_ids)

            log_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "loss_log.txt")
            with open(log_path, "a") as f:
                f.write("{}".format(total_loss) + "\n")
            self.logger.info(f"Round {round + 1} completed. Average Loss: {total_loss:.4f}")

            # Aggregate models (e.g., by sample-size weighted averaging)
            global_state_dict = None
            total_samples = sum(num_samples for _, num_samples in car_dict.values())
            for car_id, (ckpt_path, num_samples) in car_dict.items():
                state_dict = torch.load(ckpt_path)
                if global_state_dict is None:
                    global_state_dict = {key: value.clone() * (num_samples / total_samples) for key, value in state_dict.items()}
                else:
                    for key in global_state_dict:
                        global_state_dict[key] += state_dict[key] * (num_samples / total_samples)

            # Load the aggregated state dict into the model and save as the new global checkpoint
            model.load_state_dict(global_state_dict)
            global_ckpt = self.save_checkpoint(model, round + 1, keep_only_latest=False, prefix="global_")
            scheduler.step()
            self.cfg.learning_rate = scheduler.get_last_lr()[0]
        self.logger.info("Training completed.")
