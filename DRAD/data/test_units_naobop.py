import logging
import unittest

import torch

from data.naobop_dataset import TrainNaoBopDataset, EvalNaoBopDataset
from utils.dataloader import get_dataloader
from tqdm.auto import tqdm

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestBaseDataset(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def test_dataset_outputs(self):
        """Test that dataset outputs have correct shapes and types."""
        _logger.debug("Testing dataset outputs.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 256
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)

        for sample in iter(dataset):
            self.assertIn("label", sample)
            self.assertIn("car_id", sample)
            self.assertIn("charge_segment", sample)
            self.assertIn("mileage", sample)
            self.assertIn("preprocess_inputs", sample)
            self.assertIn("padded_value", sample)
            self.assertIn("timestamp", sample)
            self.assertIn("raw_voltage", sample)
            self.assertIn("raw_current", sample)
            self.assertIn("raw_min_cell_temperature", sample)
            self.assertIn("raw_max_cell_temperature", sample)
            self.assertIn("raw_min_cell_voltage", sample)
            self.assertIn("raw_max_cell_voltage", sample)
            self.assertIn("raw_soc", sample)
            self.assertIn("raw_padded_value", sample)

    def test_balance_functionality(self):
        """Test that balancing functionality works correctly."""
        _logger.debug("Testing balancing functionality.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 128
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length, skip_abnormal=False, balance=True)

        normal_count = 0
        abnormal_count = 0
        for sample in iter(dataset):
            if sample["label"].item() == 0:
                normal_count += 1
            else:
                abnormal_count += 1

        self.assertGreaterEqual(
            abnormal_count, normal_count // 2, "Abnormal samples should be at least one-half of normal samples after balancing."
        )

    def test_skip_abnormal_data(self):
        """Test that abnormal data is skipped in training dataset."""
        _logger.debug("Testing skip_abnormal_data functionality.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 256
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length, skip_abnormal=True)

        for sample in iter(dataset):
            self.assertEqual(sample["label"].item(), 0, "Abnormal data with label 1 should be skipped.")

        dataset = TrainNaoBopDataset(data_root, fold_name, max_length, skip_abnormal=False)
        has_abnormal = False
        for sample in iter(dataset):
            if sample["label"].item() == 1:
                has_abnormal = True
                break
        self.assertTrue(has_abnormal, "Abnormal data with label 1 should be included when skip_abnormal is False.")

    def test_load_dataset_with_max_length128(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=128.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 128
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)

    def test_load_dataset_with_max_length256(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=256.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 256
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)

    def test_load_dataset_with_max_length384(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=384.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 384
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)

    def test_load_dataset_with_max_length512(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=512.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 512
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)

    def test_load_dataset_with_max_length768(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=768.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 768
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)

    def test_load_dataset_with_max_length1024(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing loading dataset with max_length=1024.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_train.txt"
        max_length = 1024
        dataset = TrainNaoBopDataset(data_root, fold_name, max_length)
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        for data_dict in tqdm(dataloader):
            self.assertEqual(data_dict["preprocess_inputs"].shape[1], max_length)
            self.assertEqual(data_dict["preprocess_inputs"].shape[0], 2)  # batch size
            self.assertEqual(data_dict["raw_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_current"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_temperature"].shape[1], max_length)
            self.assertEqual(data_dict["raw_min_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_max_cell_voltage"].shape[1], max_length)
            self.assertEqual(data_dict["raw_soc"].shape[1], max_length)


class TestEvalNaoBopDataset(unittest.TestCase):
    def test_dataset_outputs(self):
        """Test that dataset outputs have correct shapes and types."""
        _logger.debug("Testing EvalNaoBopDataset outputs.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_val.txt"
        max_length = 256
        dataset = EvalNaoBopDataset(data_root, fold_name, max_length, overlap=0.0)

        for samples in tqdm(iter(dataset)):
            self.assertIsInstance(samples, list)
            for sample in samples:
                self.assertIn("label", sample)
                self.assertIn("car_id", sample)
                self.assertIn("charge_segment", sample)
                self.assertIn("mileage", sample)
                self.assertIn("preprocess_inputs", sample)
                self.assertIn("padded_value", sample)
                self.assertIn("timestamp", sample)
                self.assertIn("raw_voltage", sample)
                self.assertIn("raw_current", sample)
                self.assertIn("raw_min_cell_temperature", sample)
                self.assertIn("raw_max_cell_temperature", sample)
                self.assertIn("raw_min_cell_voltage", sample)
                self.assertIn("raw_max_cell_voltage", sample)
                self.assertIn("raw_soc", sample)
                self.assertIn("raw_padded_value", sample)

    def test_skip_abnormal_data(self):
        """Test that abnormal data is skipped in evaluation dataset."""
        _logger.debug("Testing EvalNaoBopDataset always includes all data functionality.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_val.txt"
        max_length = 256
        dataset = EvalNaoBopDataset(data_root, fold_name, max_length, overlap=0.0)

        has_abnormal = False
        for samples in tqdm(iter(dataset)):
            for sample in samples:
                if sample["label"].item() == 1:
                    has_abnormal = True
                    break
            if has_abnormal:
                break
        self.assertTrue(has_abnormal, "Abnormal data with label 1 should be included in evaluation dataset.")

    def test_overlap_functionality(self):
        """Test that overlap functionality works correctly."""
        _logger.debug("Testing EvalNaoBopDataset overlap functionality.")
        data_root = "data/battery_data/battery_brand3/"
        fold_name = "fold_0_val.txt"
        max_length = 256
        overlap = 0.25
        dataset = EvalNaoBopDataset(data_root, fold_name, max_length, overlap=overlap)

        for samples in tqdm(iter(dataset)):
            if len(samples) > 1:
                for i in range(1, len(samples)):
                    prev_sample = samples[i - 1]["timestamp"]
                    curr_sample = samples[i]["timestamp"]
                    overlap_length = int(max_length * overlap)
                    self.assertTrue(torch.equal(prev_sample[-overlap_length:], curr_sample[:overlap_length]))
