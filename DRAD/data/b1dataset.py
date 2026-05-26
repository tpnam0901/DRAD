import os
import os.path as osp
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class B1Dataset(Dataset):
    def __init__(self, data_root: str, brand_num: int, car_ids: List, mode: str = "train"):
        super(B1Dataset, self).__init__()
        self.column = torch.load(osp.join(data_root, "column.pkl"))
        self.data = []
        sample_mileage = []
        meta_data = pd.read_csv(osp.join(data_root, f"{mode}_labels.csv"))
        # Remove samples that are not in car_ids
        meta_data = meta_data[meta_data["car"].isin(car_ids)]
        # print("Loading data into memory, it may take a while...")
        for _, row in meta_data.iterrows():
            data, _ = torch.load(row["path"])
            self.data.append((data, row))
            sample_mileage.append(row["mileage"])

        self.indices = list(range(len(self.data)))

        self.max_mileage = max(sample_mileage, default=-1)
        self.min_mileage = min(sample_mileage, default=-1)

        current_dir = osp.dirname(os.path.abspath(__file__))
        self.mean = np.load(osp.join(current_dir, f"b{brand_num}_mean.npy"))
        self.std = np.load(osp.join(current_dir, f"b{brand_num}_std.npy"))
        self.min_norm = np.load(osp.join(current_dir, f"b{brand_num}_min_norm.npy"))
        self.max_norm = np.load(osp.join(current_dir, f"b{brand_num}_max_norm.npy"))

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
        return self._process_data(df, meta_data)

    def _process_data(self, df_raw: np.ndarray, meta_data: Dict) -> Dict:

        # raw_voltage = df_raw[:, self.column.index("volt")]
        # raw_current = df_raw[:, self.column.index("current")]
        # raw_soc = df_raw[:, self.column.index("soc")]
        # raw_max_cell_voltage = df_raw[:, self.column.index("max_single_volt")]
        # raw_min_cell_voltage = df_raw[:, self.column.index("min_single_volt")]
        # raw_max_cell_temperature = df_raw[:, self.column.index("max_temp")]
        # raw_min_cell_temperature = df_raw[:, self.column.index("min_temp")]
        # raw_timestamp = df_raw[:, self.column.index("timestamp")]

        df_norm = df_raw.copy()
        df_norm = (df_norm - self.mean) / np.maximum(np.maximum(1e-4, self.std), 0.1 * (self.max_norm - self.min_norm))

        normed_voltage = torch.from_numpy(df_norm[:, self.column.index("volt")]).float()
        normed_current = torch.from_numpy(df_norm[:, self.column.index("current")]).float()
        normed_min_cell_temperature = torch.from_numpy(df_norm[:, self.column.index("min_temp")]).float()
        normed_max_cell_temperature = torch.from_numpy(df_norm[:, self.column.index("max_temp")]).float()
        normed_min_cell_voltage = torch.from_numpy(df_norm[:, self.column.index("min_single_volt")]).float()
        normed_max_cell_voltage = torch.from_numpy(df_norm[:, self.column.index("max_single_volt")]).float()
        normed_soc = torch.from_numpy(df_norm[:, self.column.index("soc")]).float()
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
            # "raw_voltage": raw_voltage,
            # "raw_current": raw_current,
            # "raw_min_cell_temperature": raw_min_cell_temperature,
            # "raw_max_cell_temperature": raw_max_cell_temperature,
            # "raw_min_cell_voltage": raw_min_cell_voltage,
            # "raw_max_cell_voltage": raw_max_cell_voltage,
            # "raw_soc": raw_soc,
            # "raw_timestamp": raw_timestamp,
            "normed_voltage": normed_voltage,
            "normed_current": normed_current,
            "normed_min_cell_temperature": normed_min_cell_temperature,
            "normed_max_cell_temperature": normed_max_cell_temperature,
            "normed_min_cell_voltage": normed_min_cell_voltage,
            "normed_max_cell_voltage": normed_max_cell_voltage,
            "normed_soc": normed_soc,
            "preprocess_inputs": normed_time_series,
        }
