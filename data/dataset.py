import logging
import os
import os.path as osp
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import z_score_normalize


class BaseDataset(Dataset):
    def __init__(self, X, y, column, verbose: bool = True):

        self.X = X
        self.y = y
        self.column = column

        self.indices = list(range(len(self.X)))

        sample_mileage = []
        for i in tqdm(range(len(self.y)), disable=not verbose):
            sample_mileage.append(self.y[i][3])  # mileage
        self.max_mileage = max(sample_mileage)
        self.min_mileage = min(sample_mileage)

    def set_min_max_mileage(self, min_mileage: float, max_mileage: float):
        """
        Set the minimum and maximum mileage for the dataset.
        This is useful for normalization or other purposes.
        """
        self.min_mileage = min_mileage
        self.max_mileage = max_mileage

    def get_min_max_mileage(self):
        """
        Get the minimum and maximum mileage for the dataset.
        This is useful for normalization or other purposes.
        """
        return self.min_mileage, self.max_mileage

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        df, _ = torch.load(self.X[self.indices[index]], weights_only=False)
        df = df.astype(np.float32)

        assert (
            self.min_mileage != -1 and self.max_mileage != -1
        ), "Min and max mileage must be set before getting items for validation or testing."

        data_path, label, car, mileage, charge_segment = self.y[self.indices[index]]

        raw_voltage = df[:, self.column.index("volt")]
        raw_current = df[:, self.column.index("current")]
        raw_soc = df[:, self.column.index("soc")]
        raw_max_cell_voltage = df[:, self.column.index("max_single_volt")]
        raw_min_cell_voltage = df[:, self.column.index("min_single_volt")]
        raw_max_cell_temperature = df[:, self.column.index("max_temp")]
        raw_min_cell_temperature = df[:, self.column.index("min_temp")]
        raw_timestamp = df[:, self.column.index("timestamp")]

        normed_voltage = z_score_normalize(raw_voltage)
        normed_current = z_score_normalize(raw_current)
        normed_min_cell_temperature = z_score_normalize(raw_min_cell_temperature)
        normed_max_cell_temperature = z_score_normalize(raw_max_cell_temperature)
        normed_min_cell_voltage = z_score_normalize(raw_min_cell_voltage)
        normed_max_cell_voltage = z_score_normalize(raw_max_cell_voltage)
        normed_soc = z_score_normalize(raw_soc)
        normed_time_series = np.stack(
            [
                normed_soc,
                normed_current,
                normed_min_cell_temperature,
                normed_max_cell_voltage,
                normed_min_cell_voltage,
                normed_max_cell_temperature,
                normed_voltage,
            ],
            axis=1,
        )

        label = torch.tensor(label, dtype=torch.long)
        car = torch.tensor(int(car), dtype=torch.long)
        normed_mileage = (torch.tensor(mileage, dtype=torch.float32) - self.min_mileage) / (self.max_mileage - self.min_mileage)
        charge_segment = torch.tensor(charge_segment, dtype=torch.long)

        return {
            # "data_path": data_path,
            "label": label,
            "car": car,
            "normed_mileage": normed_mileage,
            "charge_segment": charge_segment,
            "raw_voltage": torch.tensor(raw_voltage, dtype=torch.float32),
            "raw_current": torch.tensor(raw_current, dtype=torch.float32),
            "raw_min_cell_temperature": torch.tensor(raw_min_cell_temperature, dtype=torch.float32),
            "raw_max_cell_temperature": torch.tensor(raw_max_cell_temperature, dtype=torch.float32),
            "raw_min_cell_voltage": torch.tensor(raw_min_cell_voltage, dtype=torch.float32),
            "raw_max_cell_voltage": torch.tensor(raw_max_cell_voltage, dtype=torch.float32),
            "raw_soc": torch.tensor(raw_soc, dtype=torch.float32),
            "raw_timestamp": torch.tensor(raw_timestamp, dtype=torch.float32),
            "normed_voltage": torch.tensor(normed_voltage, dtype=torch.float32),
            "normed_current": torch.tensor(normed_current, dtype=torch.float32),
            "normed_min_cell_temperature": torch.tensor(normed_min_cell_temperature, dtype=torch.float32),
            "normed_max_cell_temperature": torch.tensor(normed_max_cell_temperature, dtype=torch.float32),
            "normed_min_cell_voltage": torch.tensor(normed_min_cell_voltage, dtype=torch.float32),
            "normed_max_cell_voltage": torch.tensor(normed_max_cell_voltage, dtype=torch.float32),
            "normed_soc": torch.tensor(normed_soc, dtype=torch.float32),
            "normed_time_series": torch.tensor(normed_time_series, dtype=torch.float32),
        }


def build_dataset(
    data_root: str,
    brand_num: int = 1,
    mode: str = "train",
    car_id: int = -1,
    logger: logging.Logger = logging.getLogger(__name__),
    verbose: bool = True,
    train_include: bool = False,
) -> BaseDataset:

    data_root = os.path.join(data_root, "battery_brand{}".format(brand_num))
    metadata_path = os.path.join(data_root, "drv_{}_labels.csv".format(mode))
    column = torch.load(osp.join(data_root, "column.pkl"))
    assert os.path.exists(metadata_path), "Metadata path does not exist: {}".format(metadata_path)
    df = pd.read_csv(metadata_path)
    X = []
    y = []

    if verbose:
        logger.info("Building dataset from {}".format(data_root))
    for _, row in tqdm(df.iterrows(), disable=not verbose):
        # Row: filename, label, car, mileage, charge_segment
        data_path = os.path.join(data_root, "train", row["filename"])
        if not os.path.exists(data_path):
            data_path = os.path.join(data_root, "test", row["filename"])
        if not os.path.exists(data_path):
            data_path = os.path.join(data_root, "data", row["filename"])
        assert os.path.exists(data_path), "Data path does not exist: {}".format(data_path)
        if car_id != -1 and int(row["car"]) != car_id:
            continue
        X.append(data_path)
        y.append((data_path, row["label"], row["car"], row["mileage"], row["charge_segment"]))

    if train_include and mode != "train":
        metadata_path = os.path.join(data_root, "drv_{}_labels.csv".format(mode))
        df = pd.read_csv(metadata_path)
        # If train_include is True, include all data for training, even if car_id is specified
        for _, row in tqdm(df.iterrows(), disable=not verbose):
            data_path = os.path.join(data_root, "train", row["filename"])
            if not os.path.exists(data_path):
                data_path = os.path.join(data_root, "test", row["filename"])
            if not os.path.exists(data_path):
                data_path = os.path.join(data_root, "data", row["filename"])
            assert os.path.exists(data_path), "Data path does not exist: {}".format(data_path)
            if car_id != -1 and int(row["car"]) == car_id:
                continue
            X.append(data_path)
            y.append((data_path, row["label"], row["car"], row["mileage"], row["charge_segment"]))

    dataset = BaseDataset(X, y, column, verbose=verbose)

    return dataset
