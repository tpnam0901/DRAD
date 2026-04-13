import logging
import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import (
    chunk_tensor_with_overlap,
    random_crop_tensor,
    z_score_normalize,
)


class BaseDataset(Dataset):
    def __init__(self, data_root: str, fold_name: str, max_length: int, car_ids: List):
        super(BaseDataset, self).__init__()
        assert max_length == 128, "Only support max_length of 128 for now"
        self.column = torch.load(osp.join(data_root, "column.pkl"))
        self.max_length = max_length
        with open(osp.join(data_root, fold_name), "r") as f:
            filenames = f.readlines()
        # print("Loading data into memory, it may take a while...")

        self.data = []
        sample_mileage = []
        for i, filename in enumerate(filenames):
            filename = filename.strip()
            data, meta_data = torch.load(osp.join(data_root, "data_by_segments", filename))
            sample_mileage.append(meta_data["mileage"])
            if int(meta_data["car"]) not in car_ids:
                continue
            self.data.append((data, meta_data))

        self.indices = list(range(len(self.data)))

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
        df, meta_data = self.data[self.indices[index]]
        df = df.astype(np.float32)
        df = random_crop_tensor(df, self.max_length, dim=0)

        return self._process_data(df, meta_data)

    def _process_data(self, df: np.ndarray, meta_data: Dict) -> Dict:
        assert (
            self.min_mileage != -1 and self.max_mileage != -1
        ), "Min and max mileage must be set before getting items for validation or testing."

        # data_path, label, car, mileage, charge_segment = self.y[self.indices[index]]

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
        normed_time_series = torch.stack(
            [
                normed_soc,
                normed_current,
                normed_min_cell_temperature,
                normed_max_cell_voltage,
                normed_min_cell_voltage,
                normed_max_cell_temperature,
                normed_voltage,
            ],
            dim=1,
        )

        label = torch.tensor(int(meta_data["label"]), dtype=torch.long)
        car = torch.tensor(int(meta_data["car"]), dtype=torch.long)
        normed_mileage = (torch.tensor(float(meta_data["mileage"]), dtype=torch.float32) - self.min_mileage) / (
            self.max_mileage - self.min_mileage
        )
        charge_segment = torch.tensor(int(meta_data["charge_segment"]), dtype=torch.long)

        return {
            # "data_path": data_path,
            "label": label,
            "car": car,
            "normed_mileage": normed_mileage,
            "charge_segment": charge_segment,
            "raw_voltage": raw_voltage.float(),
            "raw_current": raw_current.float(),
            "raw_min_cell_temperature": raw_min_cell_temperature.float(),
            "raw_max_cell_temperature": raw_max_cell_temperature.float(),
            "raw_min_cell_voltage": raw_min_cell_voltage.float(),
            "raw_max_cell_voltage": raw_max_cell_voltage.float(),
            "raw_soc": raw_soc.float(),
            "raw_timestamp": raw_timestamp.float(),
            "normed_voltage": normed_voltage.float(),
            "normed_current": normed_current.float(),
            "normed_min_cell_temperature": normed_min_cell_temperature.float(),
            "normed_max_cell_temperature": normed_max_cell_temperature.float(),
            "normed_min_cell_voltage": normed_min_cell_voltage.float(),
            "normed_max_cell_voltage": normed_max_cell_voltage.float(),
            "normed_soc": normed_soc.float(),
            "normed_time_series": normed_time_series.float(),
        }


class EvalBaseDataset(BaseDataset):
    def __init__(self, data_root: str, fold_name: str, max_length: int, car_ids: List[int]):
        super(EvalBaseDataset, self).__init__(data_root, fold_name, max_length, car_ids)
        self.overlap = 0.0

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        data_raw, meta_data = self.data[index]
        data_chunks = chunk_tensor_with_overlap(data_raw, self.max_length, overlap=self.overlap, dim=0)
        return_data = []
        for data in data_chunks:
            processed_data = self._process_data(data, meta_data)
            return_data.append(processed_data)
        return return_data


def build_dataset(
    data_root: str,
    brand_num: int = 3,
    mode: str = "train",
    car_ids: List[int] = [],
    fold_num: int = 0,
    max_length: int = 128,
) -> BaseDataset:

    data_root = os.path.join(data_root, "battery_brand{}".format(brand_num))
    if mode != "train":
        return EvalBaseDataset(
            data_root,
            f"fold_{fold_num}_{mode}.txt",
            max_length,
            car_ids=car_ids,
        )
    return BaseDataset(
        data_root,
        f"fold_{fold_num}_{mode}.txt",
        max_length,
        car_ids=car_ids,
    )
