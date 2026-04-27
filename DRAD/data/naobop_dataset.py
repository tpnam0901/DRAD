import os.path as osp
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

from data.basedataset import BaseDataset
from utils.data import (
    chunk_tensor_with_overlap,
    min_max_mileage_normalize,
    padding_to_max_length,
    random_crop_tensor,
    z_score_normalize,
)


class TrainNaoBopDataset(BaseDataset):
    def __init__(self, data_root: str, fold_name: str, max_length: int, car_id: int):
        super(BaseDataset, self).__init__()
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
            if int(meta_data["car"]) != car_id:
                continue
            self.data.append((data, meta_data))

        self.indices = list(range(len(self.data)))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data, meta_data = self.data[self.indices[index]]
        data = random_crop_tensor(data, self.max_length, dim=0)
        return self._process_data(data, meta_data)

    def _process_data(self, data: torch.Tensor, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = data[:, self.column.index("timestamp")]
        raw_voltage = data[:, self.column.index("volt")]
        raw_current = data[:, self.column.index("current")]
        raw_min_cell_temperature = data[:, self.column.index("min_temp")]
        raw_max_cell_temperature = data[:, self.column.index("max_temp")]
        raw_min_cell_voltage = data[:, self.column.index("min_single_volt")]
        raw_max_cell_voltage = data[:, self.column.index("max_single_volt")]
        raw_soc = data[:, self.column.index("soc")]

        if isinstance(raw_voltage, np.ndarray):
            timestamp = torch.from_numpy(timestamp)
            raw_voltage = torch.from_numpy(raw_voltage)
            raw_current = torch.from_numpy(raw_current)
            raw_min_cell_temperature = torch.from_numpy(raw_min_cell_temperature)
            raw_max_cell_temperature = torch.from_numpy(raw_max_cell_temperature)
            raw_min_cell_voltage = torch.from_numpy(raw_min_cell_voltage)
            raw_max_cell_voltage = torch.from_numpy(raw_max_cell_voltage)
            raw_soc = torch.from_numpy(raw_soc)

        normed_voltage: torch.Tensor = z_score_normalize(raw_voltage)
        normed_current: torch.Tensor = z_score_normalize(raw_current)
        normed_min_cell_temperature: torch.Tensor = z_score_normalize(raw_min_cell_temperature)
        normed_max_cell_temperature: torch.Tensor = z_score_normalize(raw_max_cell_temperature)
        normed_min_cell_voltage: torch.Tensor = z_score_normalize(raw_min_cell_voltage)
        normed_max_cell_voltage: torch.Tensor = z_score_normalize(raw_max_cell_voltage)
        normed_soc: torch.Tensor = z_score_normalize(raw_soc)

        padded_value = -2.0
        dim = 0
        padded_voltage = padding_to_max_length(normed_voltage, self.max_length, dim=dim, padding_value=padded_value)
        padded_current = padding_to_max_length(normed_current, self.max_length, dim=dim, padding_value=padded_value)
        padded_min_cell_temperature = padding_to_max_length(
            normed_min_cell_temperature, self.max_length, dim=dim, padding_value=padded_value
        )
        padded_max_cell_temperature = padding_to_max_length(
            normed_max_cell_temperature, self.max_length, dim=dim, padding_value=padded_value
        )
        padded_min_cell_voltage = padding_to_max_length(normed_min_cell_voltage, self.max_length, dim=dim, padding_value=padded_value)
        padded_max_cell_voltage = padding_to_max_length(normed_max_cell_voltage, self.max_length, dim=dim, padding_value=padded_value)
        padded_soc = padding_to_max_length(normed_soc, self.max_length, dim=dim, padding_value=padded_value)

        preprocess_inputs = torch.stack(
            [
                padded_soc,
                padded_current,
                padded_min_cell_temperature,
                padded_max_cell_temperature,
                padded_min_cell_voltage,
                padded_max_cell_voltage,
                padded_voltage,
            ],
            dim=1,
        )  # Shape: (max_length, 7 features)

        padded_raw_value = -999.0
        padded_raw_voltage = padding_to_max_length(raw_voltage, self.max_length, dim=dim, padding_value=padded_raw_value)
        padded_raw_current = padding_to_max_length(raw_current, self.max_length, dim=0, padding_value=padded_raw_value)
        padded_raw_min_cell_temperature = padding_to_max_length(
            raw_min_cell_temperature, self.max_length, dim=dim, padding_value=padded_raw_value
        )
        padded_raw_max_cell_temperature = padding_to_max_length(
            raw_max_cell_temperature, self.max_length, dim=dim, padding_value=padded_raw_value
        )
        padded_raw_min_cell_voltage = padding_to_max_length(raw_min_cell_voltage, self.max_length, dim=dim, padding_value=padded_raw_value)
        padded_raw_max_cell_voltage = padding_to_max_length(raw_max_cell_voltage, self.max_length, dim=dim, padding_value=padded_raw_value)
        padded_raw_soc = padding_to_max_length(raw_soc, self.max_length, dim=dim, padding_value=padded_raw_value)
        padded_timestamp = padding_to_max_length(timestamp, self.max_length, dim=dim, padding_value=padded_raw_value)

        return {
            "label": torch.tensor(int(meta_data["label"]), dtype=torch.long),
            "car_id": meta_data["car"],
            "charge_segment": meta_data["charge_segment"],
            "mileage": torch.tensor(meta_data["mileage"], dtype=torch.float32),
            "norm_mileage": min_max_mileage_normalize(torch.tensor(meta_data["mileage"], dtype=torch.float32)),
            "preprocess_inputs": preprocess_inputs,
            "padded_value": padded_value,
            "timestamp": padded_timestamp,
            "raw_voltage": padded_raw_voltage,
            "raw_current": padded_raw_current,
            "raw_min_cell_temperature": padded_raw_min_cell_temperature,
            "raw_max_cell_temperature": padded_raw_max_cell_temperature,
            "raw_min_cell_voltage": padded_raw_min_cell_voltage,
            "raw_max_cell_voltage": padded_raw_max_cell_voltage,
            "raw_soc": padded_raw_soc,
            "raw_padded_value": padded_raw_value,
            "normed_voltage": padded_voltage,
            "normed_current": padded_current,
            "normed_min_cell_temperature": padded_min_cell_temperature,
            "normed_max_cell_temperature": padded_max_cell_temperature,
            "normed_min_cell_voltage": padded_min_cell_voltage,
            "normed_max_cell_voltage": padded_max_cell_voltage,
            "normed_soc": padded_soc,
        }

    def __len__(self):
        return len(self.indices)


class EvalNaoBopDataset(TrainNaoBopDataset):
    def __init__(self, data_root: str, fold_name: str, max_length: int, car_id: int):
        super(EvalNaoBopDataset, self).__init__(data_root, fold_name, max_length, car_id)
        self.overlap = 0.0

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        data_raw, meta_data = self.data[index]
        data_chunks = chunk_tensor_with_overlap(data_raw, self.max_length, overlap=self.overlap, dim=0)
        return_data = []
        for data in data_chunks:
            processed_data = self._process_data(data, meta_data)
            return_data.append(processed_data)
        return return_data
