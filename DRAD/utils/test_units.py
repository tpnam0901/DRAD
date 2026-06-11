import logging
import math
import unittest

import torch

from configs.base import Config
from data.basedataset import BaseDataset
from utils import schedulers
from utils.dataloader import get_dataloader
from utils import data

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestDataloader(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def setUp(self):
        """Set up test fixtures."""
        x = [i for i in range(10)]
        y = [i % 2 for i in range(10)]
        self.dataset = BaseDataset(x, y)

    def test_dataloader_batch_size(self):
        """Test that dataloader returns correct batch size."""
        _logger.debug("Testing dataloader batch size.")
        dataloader = get_dataloader(self.dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch["x"]), 4)
        self.assertEqual(len(batch["y"]), 4)

    def test_dataloader_iteration(self):
        """Test that dataloader can iterate through the dataset."""
        _logger.debug("Testing dataloader iteration.")
        dataloader = get_dataloader(self.dataset, batch_size=2)
        total_samples = 0
        for batch in dataloader:
            total_samples += len(batch["x"])
        self.assertEqual(total_samples, len(self.dataset))

    def test_dataloader_drop_last(self):
        """Test that dataloader drops last incomplete batch when drop_last is True."""
        _logger.debug("Testing dataloader drop_last functionality.")
        dataloader = get_dataloader(self.dataset, batch_size=3, drop_last=True)
        total_samples = 0
        for batch in dataloader:
            total_samples += len(batch["x"])
        self.assertEqual(total_samples, 9)  # 10 samples with batch size 3 drops last sample


class TestScheduler(unittest.TestCase):
    """Test cases for different learning rate schedulers."""

    def setUp(self):
        """Set up test fixtures."""

        self.cfg = Config()
        self.cfg.unlock()
        self.cfg.learning_rate = 0.1
        self.cfg.lock()

        # Create a simple model and optimizer for testing
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    def test_step_lr_scheduler(self):
        """Test StepLR scheduler decreases learning rate at specified step."""
        _logger.debug("Testing StepLR scheduler.")

        self.cfg.unlock()
        self.cfg.lr_step_size = 5
        self.cfg.lr_step_gamma = 0.5
        self.cfg.lock()

        scheduler = schedulers.StepLR(self.optimizer, self.cfg)
        initial_lr = self.optimizer.param_groups[0]["lr"]

        # Track learning rates
        lrs = [initial_lr]
        for epoch in range(15):
            scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])

        # Verify LR decreases at step_size intervals
        self.assertAlmostEqual(lrs[0], 0.1, places=5)
        self.assertAlmostEqual(lrs[5], 0.05, places=5)  # After 5 epochs: 0.1 * 0.5
        self.assertAlmostEqual(lrs[10], 0.025, places=5)  # After 10 epochs: 0.1 * 0.5^2
        self.assertAlmostEqual(lrs[15], 0.0125, places=5)  # After 15 epochs: 0.1 * 0.5^3

        _logger.debug(f"StepLR learning rates: {lrs}")

    def test_multi_step_lr_scheduler(self):
        _logger.debug(f"Testing MultiStepLR scheduler is not yet implemented.")

    def test_exponential_lr_scheduler(self):
        _logger.debug(f"Testing ExponentialLR scheduler is not yet implemented.")

    def test_cosine_annealing_lr_scheduler(self):
        """Test CosineAnnealingLR scheduler decreases learning rate following cosine annealing."""
        _logger.debug("Testing CosineAnnealingLR scheduler.")

        self.cfg.unlock()
        self.cfg.lr_T_max = 10
        self.cfg.lr_eta_min = 0.0
        self.cfg.lock()

        scheduler = schedulers.CosineAnnealingLR(self.optimizer, self.cfg)
        initial_lr = self.optimizer.param_groups[0]["lr"]

        lrs = [initial_lr]
        for epoch in range(10):
            scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(lrs[0], 0.1, places=5)  # Initial LR
        lr_1 = 0.0 + (0.1 - 0.0) * (1 + math.cos(math.pi * 1 / 10)) / 2
        self.assertAlmostEqual(lrs[1], lr_1, places=5)
        lr_2 = 0.0 + (0.1 - 0.0) * (1 + math.cos(math.pi * 2 / 10)) / 2
        self.assertAlmostEqual(lrs[2], lr_2, places=5)
        lr_5 = 0.0 + (0.1 - 0.0) * (1 + math.cos(math.pi * 5 / 10)) / 2
        self.assertAlmostEqual(lrs[5], lr_5, places=5)
        self.assertAlmostEqual(lrs[10], 0.0, places=5)

        _logger.debug(f"CosineAnnealingLR learning rates: {lrs}")

    def test_reduce_lr_on_plateau_scheduler(self):
        _logger.debug(f"Testing ReduceLROnPlateau scheduler is not yet implemented.")

    def test_cosine_annealing_warm_restarts_scheduler(self):
        _logger.debug(f"Testing CosineAnnealingWarmRestarts scheduler is not yet implemented.")

    def test_identity_scheduler(self):
        _logger.debug(f"Testing IdentityScheduler is not yet implemented.")


class TestData(unittest.TestCase):
    """Test cases for data loading and processing."""

    def test_min_max_mileage_normalize(self):
        """Test min-max normalization for mileage data."""
        _logger.debug("Testing min_max_mileage_normalize function.")
        data_mileage = torch.tensor([0.0, 22181.18826, 44362.37652])
        normalized_mileage = data.min_max_mileage_normalize(data_mileage)
        expected_mileage = data_mileage / 44362.376520000005
        self.assertTrue(torch.allclose(normalized_mileage, expected_mileage, atol=1e-5))

    def test_min_max_normalization_dataset(self):
        """Test max normalization function."""
        _logger.debug("Testing max_normalize function.")
        max_values = {
            "volt": 46.172,
            "current": 180.78,
            "soc": 100.0,
            "max_single_volt": 4.2309,
            "min_single_volt": 4.2151,
            "max_temp": 39.0,
            "min_temp": 36.0,
        }
        min_values = {
            "volt": 36.20217999999999,
            "current": -180.78,
            "soc": 0.0,
            "max_single_volt": 3.4252,
            "min_single_volt": 3.3384,
            "max_temp": -4.0,
            "min_temp": -4.0,
        }
        data_volt = torch.tensor([0.0, 23.086, 46.172])
        normalized_volt = data.min_max_normalize_dataset(data_volt, type_data="volt")
        expected_volt = (data_volt - min_values["volt"]) / (max_values["volt"] - min_values["volt"])
        self.assertTrue(torch.allclose(normalized_volt, expected_volt, atol=1e-5))
        data_current = torch.tensor([0.0, 90.39, 180.78])
        normalized_current = data.min_max_normalize_dataset(data_current, type_data="current")
        expected_current = (data_current - min_values["current"]) / (max_values["current"] - min_values["current"])
        self.assertTrue(torch.allclose(normalized_current, expected_current, atol=1e-5))
        data_soc = torch.tensor([0.0, 50.0, 100.0])
        normalized_soc = data.min_max_normalize_dataset(data_soc, type_data="soc")
        expected_soc = (data_soc - min_values["soc"]) / (max_values["soc"] - min_values["soc"])
        self.assertTrue(torch.allclose(normalized_soc, expected_soc, atol=1e-5))
        data_max_single_volt = torch.tensor([0.0, 2.11545, 4.2309])
        normalized_max_single_volt = data.min_max_normalize_dataset(data_max_single_volt, type_data="max_single_volt")
        expected_max_single_volt = (data_max_single_volt - min_values["max_single_volt"]) / (
            max_values["max_single_volt"] - min_values["max_single_volt"]
        )
        self.assertTrue(torch.allclose(normalized_max_single_volt, expected_max_single_volt, atol=1e-5))
        data_min_single_volt = torch.tensor([0.0, 2.10755, 4.2151])
        normalized_min_single_volt = data.min_max_normalize_dataset(data_min_single_volt, type_data="min_single_volt")
        expected_min_single_volt = (data_min_single_volt - min_values["min_single_volt"]) / (
            max_values["min_single_volt"] - min_values["min_single_volt"]
        )
        self.assertTrue(torch.allclose(normalized_min_single_volt, expected_min_single_volt, atol=1e-5))
        data_max_temp = torch.tensor([0.0, 19.5, 39.0])
        normalized_max_temp = data.min_max_normalize_dataset(data_max_temp, type_data="max_temp")
        expected_max_temp = (data_max_temp - min_values["max_temp"]) / (max_values["max_temp"] - min_values["max_temp"])
        self.assertTrue(torch.allclose(normalized_max_temp, expected_max_temp, atol=1e-5))
        data_min_temp = torch.tensor([0.0, 18.0, 36.0])
        normalized_min_temp = data.min_max_normalize_dataset(data_min_temp, type_data="min_temp")
        expected_min_temp = (data_min_temp - min_values["min_temp"]) / (max_values["min_temp"] - min_values["min_temp"])
        self.assertTrue(torch.allclose(normalized_min_temp, expected_min_temp, atol=1e-5))

    def test_padding_to_max_length(self):
        """Test padding function to ensure correct output shape."""
        _logger.debug("Testing padding_to_max_length function.")
        data_tensor = torch.randn(50, 10)  # Original tensor of shape (50, 10)
        max_length = 100
        padded_tensor = data.padding_to_max_length(data_tensor, max_length, dim=0, padding_value=0)
        self.assertEqual(padded_tensor.shape[0], max_length)
        self.assertEqual(padded_tensor.shape[1], 10)
        self.assertTrue(torch.all(padded_tensor[50:, :] == 0))  # Check that padding is zero

        data_tensor = torch.randn(10, 50)  # Original tensor of shape (10, 50)
        max_length = 80
        padded_tensor = data.padding_to_max_length(data_tensor, max_length, dim=1, padding_value=1)
        self.assertEqual(padded_tensor.shape[0], 10)
        self.assertEqual(padded_tensor.shape[1], max_length)
        self.assertTrue(torch.all(padded_tensor[:, 50:] == 1))  # Check that padding is one

    def test_random_crop_tensor(self):
        """Test random cropping function to ensure correct output shape."""
        _logger.debug("Testing random_crop_tensor function.")
        notEqual = False
        for _ in range(50):
            data_tensor = torch.randn(100, 10)  # Original tensor of shape (100, 10)
            crop_size = 30
            cropped_tensor = data.random_crop_tensor(data_tensor, crop_size, dim=0)
            cropped_tensor2 = data.random_crop_tensor(data_tensor, crop_size, dim=0)
            self.assertEqual(cropped_tensor.shape[0], crop_size)
            self.assertEqual(cropped_tensor.shape[1], 10)
            if torch.sum(cropped_tensor - cropped_tensor2).item() != 0:  # Ensure randomness
                notEqual = True
                break
        notEqual = False
        for _ in range(50):
            data_tensor = torch.randn(10, 100)  # Original tensor of shape (10, 100)
            crop_size = 40
            cropped_tensor = data.random_crop_tensor(data_tensor, crop_size, dim=1)
            cropped_tensor2 = data.random_crop_tensor(data_tensor, crop_size, dim=1)
            self.assertEqual(cropped_tensor.shape[0], 10)
            print(cropped_tensor.shape, crop_size)
            self.assertEqual(cropped_tensor.shape[1], crop_size)
            if torch.sum(cropped_tensor - cropped_tensor2).item() != 0:  # Ensure randomness
                notEqual = True
                break
        self.assertTrue(notEqual)

    def test_chunk_tensor_with_overlap(self):
        """Test chunking with overlap function to ensure correct number of chunks."""
        _logger.debug("Testing chunk_tensor_with_overlap function.")
        data_tensor = torch.randn(250, 10)  # Original tensor of shape (250, 10)
        chunk_size = 64
        overlap = 0.25
        chunks = data.chunk_tensor_with_overlap(data_tensor, chunk_size, overlap, dim=0)
        step_size = int(chunk_size * (1 - overlap))
        expected_num_chunks = math.ceil((250 - chunk_size) / step_size) + 1
        self.assertEqual(len(chunks), expected_num_chunks)
        for i, chunk in enumerate(chunks):
            if i < expected_num_chunks - 1:
                self.assertEqual(chunk.shape[0], chunk_size)
            else:
                self.assertTrue(chunk.shape[0] <= chunk_size)  # Last chunk size can be smaller

    def test_z_score_normalization(self):
        """Test z-score normalization function."""
        _logger.debug("Testing z_score_normalization function.")
        data_tensor = torch.randn(100, 10) * 5 + 10  # Data with mean ~10 and std ~5
        normalized_data: torch.Tensor = data.z_score_normalize(data_tensor, dim=0)
        mean = normalized_data.mean(dim=0)
        std = normalized_data.std(dim=0)
        self.assertTrue(torch.allclose(mean, torch.zeros(10), atol=1e-5))
        self.assertTrue(torch.allclose(std, torch.ones(10), atol=1e-5))

    def test_min_max_normalization(self):
        """Test min-max normalization function."""
        _logger.debug("Testing min_max_normalization function.")
        data_tensor = torch.randn(100, 10) * 5 + 10  # Data with mean ~10 and std ~5
        normalized_data: torch.Tensor = data.min_max_normalize(data_tensor, dim=0)
        min_vals = normalized_data.min(dim=0).values
        max_vals = normalized_data.max(dim=0).values
        self.assertTrue(torch.allclose(min_vals, torch.zeros(10), atol=1e-5))
        self.assertTrue(torch.allclose(max_vals, torch.ones(10), atol=1e-5))
