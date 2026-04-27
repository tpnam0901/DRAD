import os.path as osp
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.optim import SGD
from tqdm.auto import tqdm

from data.naobop_dataset import TrainNaoBopDataset as BaseDataset
from engine.train_drv import TrainEngine as BaseEngine
from utils.dataloader import get_dataloader
from utils.schedulers import CosineAnnealingLR


class TrainNaoBopDataset(BaseDataset):
    def __init__(self, data_root: str, fold_name: str, max_length: int, car_ids: List):
        assert max_length % 64 == 0, "max_length should be multiple of 64"
        self.column = torch.load(osp.join(data_root, "column.pkl"))
        self.max_length = max_length
        with open(osp.join(data_root, fold_name), "r") as f:
            filenames = f.readlines()
        # print("Loading data into memory, it may take a while...")

        self.data = []
        for i, filename in enumerate(filenames):
            filename = filename.strip()
            data, meta_data = torch.load(osp.join(data_root, "data_by_segments", filename))
            if int(meta_data["car"]) not in car_ids:
                continue
            self.data.append((data, meta_data))

        self.indices = list(range(len(self.data)))


class TrainEngine(BaseEngine):
    def load_dataset(
        self,
        data_root: str,
        fold_num: int,
        max_length: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        car_ids: List[int] = [],
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
            car_ids=car_ids,
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

    def run(self):
        """Run the training process."""

        # car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand3/label/all_label.csv")
        # car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand2/fold_label.csv")
        car_info = pd.read_csv("/home/phuongnam/DistributedEVTest/data/battery_data/battery_brand1/fold_label.csv")
        car_normal_ids = car_info[car_info["label"] == 0]["car"].unique().tolist()

        datasets = self.load_dataset(
            self.cfg.data_root,
            self.cfg.fold_num,
            self.cfg.max_length,
            self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            car_ids=car_normal_ids,
        )
        train_loader = datasets["train_loader"]
        model = self.build_model()
        optimizer = SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.99, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, self.cfg)
        log_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "loss_log.txt")
        with tqdm(total=self.cfg.num_epochs, unit="epoch") as pbar:
            for epoch in range(self.cfg.num_epochs):
                loss_epoch = self.train_epoch(model, train_loader, optimizer, scheduler, prefix=f"all_normal_")
                self.save_checkpoint(model, epoch + 1, keep_only_latest=True, prefix=f"all_normal_")
                pbar.set_description(f"All Normal Training - Loss: {loss_epoch:.4f}")
                pbar.update(1)
                with open(log_path, "a") as f:
                    f.write("{}".format(loss_epoch) + "\n")
        self.logger.info("Training completed.")
