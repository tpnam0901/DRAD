import logging
import tempfile
import unittest

import mlflow

import torch

from engine.base import BaseEngine

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestBaseEngine(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def test_base_engine_without_log_path(self):
        """Test BaseEngine initialization without log path."""
        _logger.debug("Testing BaseEngine initialization without log path.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        engine = DummyClass()
        self.assertIsNotNone(engine.logger)
        # Test logging prints to console
        _logger.setLevel(logging.DEBUG)
        with self.assertLogs(engine.logger, level="DEBUG") as log:
            engine.logger.debug("Test log entry.")
            self.assertIn("Test log entry.", log.output[0])
        _logger.setLevel(logging.root.level)

    def test_base_engine_with_log_path_and_handler(self):
        """Test BaseEngine initialization with log path."""
        _logger.debug("Testing BaseEngine initialization with log path.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        with tempfile.NamedTemporaryFile(delete=True) as temp_log_file:
            engine = DummyClass(log_path=temp_log_file.name)
            self.assertIsNotNone(engine.logger)
            handlers = engine.logger.handlers
            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in handlers))
            # Test that the log file is created when logging
            _logger.setLevel(logging.DEBUG)
            engine.logger.debug("Test log entry.")
            with open(temp_log_file.name, "r") as f:
                log_contents = f.read()
                self.assertIn("Test log entry.", log_contents)
            _logger.setLevel(logging.root.level)

    def test_setup_mlflow(self):
        """Test MLflow setup in BaseEngine."""
        _logger.debug("Testing MLflow setup in BaseEngine.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        engine = DummyClass()
        engine.setup_mlflow(run_name="test_run", experiment_name="test_experiment")
        self.assertIsNotNone(engine.mlflow_id)
        self.assertEqual(engine.mlflow_run_name, "test_run")
        # Test logging with mlflow id and run name
        with mlflow.start_run(run_name=engine.mlflow_run_name, run_id=engine.mlflow_id) as run:
            self.assertEqual(run.info.run_id, engine.mlflow_id)
            self.assertEqual(run.info.run_name, engine.mlflow_run_name)


class TestTrainEngine(unittest.TestCase):
    """Test cases for TrainEngine."""

    def test_load_dataset(self):
        """Test loading of training dataset in TrainEngine."""
        from configs.base import Config
        from engine.train import TrainEngine

        _logger.debug("Testing loading of training dataset in TrainEngine.")
        cfg = Config()
        engine = TrainEngine(cfg)
        datasets = engine.load_dataset(
            cfg.data_root,
            cfg.fold_num,
            cfg.max_length,
            cfg.overlap,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        self.assertIn("train_dataset", datasets)
        self.assertIn("val_dataset", datasets)
        self.assertIn("train_loader", datasets)
        train_batch = next(iter(datasets["train_loader"]))
        self.assertEqual(train_batch["preprocess_inputs"].shape[0], 2)  # Batch size

    def test_calculate_loss(self):
        """Test loss calculation in TrainEngine."""
        import torch

        from configs.example import Config as ExampleConfig
        from engine.train import TrainEngine

        _logger.debug("Testing loss calculation in TrainEngine.")
        cfg = ExampleConfig()
        engine = TrainEngine(cfg)

        # Create dummy data
        targets = {"preprocess_inputs": torch.tensor([[[0.0, 0.0, 2.0, 4.0, 3.0]]]).float().cuda()}
        logits = {
            "log_p": torch.tensor([[[1.5, 3.5, 5.5]]]).float().cuda(),
            "mean": torch.tensor([[[2.0, 4.0, 6.0]]]).float().cuda(),
            "log_v": torch.tensor([[[0.1, 0.1, 0.1]]]).float().cuda(),
        }

        # Calculate loss
        loss_dict = engine.calculate_loss(logits, targets)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("nll_loss", loss_dict)
        self.assertIn("kl_loss", loss_dict)
        # In this case, SmoothL1Loss (nll_loss, beta=1.0)
        # For y, it will remove first two dimensions so only [2.0,4.0,3.0] remains
        # out_1 = |1.5-2.0| = 0.5 < beta -> loss_1 = 0.5*(1.5-2.0)^2/beta = 0.125
        # out_2 = |3.5-4.0| = 0.5 < beta -> loss_2 = 0.5*(3.5-4.0)^2/beta = 0.125
        # out_3 = |5.5-3.0| = 2.5 > beta -> loss_3 = |5.5-3.0| - 0.5*beta = 2.0
        # total nll_loss = (0.125 + 0.125 + 2.0)/3 = 0.75
        self.assertAlmostEqual(loss_dict["nll_loss"].item(), 0.75, places=4)

        # KL loss
        # Mean pow 2 = [4.0,16.0,36.0]
        # log_v exp = [exp(0.1), exp(0.1), exp(0.1)] = [1.1051709180756477, 1.1051709180756477, 1.1051709180756477]
        # 1+ log_v - mean^2 - exp(log_v) = [1+0.1-4.0-1.1051709180756477, 1+0.1-16.0-1.1051709180756477, 1+0.1-36.0-1.1051709180756477]
        # = [-4.005170918075647, -16.00517091807565, -36.00517091807565]
        # sum = -56.01551275422695
        # kl_loss = -0.5 * sum = 28.007756377113473
        self.assertAlmostEqual(loss_dict["kl_loss"].item(), 28.007756377113473, places=4)

        # Check total loss with kl_weight
        anneal0 = 0.1
        x0 = 500
        nll_weight = 10.0
        kl_weight = anneal0 * min(1, engine.step / x0)
        expected_total_loss = nll_weight * loss_dict["nll_loss"].item() + kl_weight * loss_dict["kl_loss"].item() / logits["log_p"].shape[0]
        self.assertAlmostEqual(loss_dict["total_loss"].item(), expected_total_loss, places=4)

        # Increase step and check kl_weight changes
        engine.step = 600
        loss_dict = engine.calculate_loss(logits, targets)
        kl_weight = anneal0 * min(1, engine.step / x0)
        expected_total_loss = nll_weight * loss_dict["nll_loss"].item() + kl_weight * loss_dict["kl_loss"].item() / logits["log_p"].shape[0]
        self.assertAlmostEqual(loss_dict["total_loss"].item(), expected_total_loss, places=4)


class TestEvaluateEngine(unittest.TestCase):
    """Test cases for EvaluateEngine."""

    def test_calculate_mse_loss(self):
        """Test MSE loss calculation in EvaluateEngine."""
        import torch

        from configs.example import Config as ExampleConfig
        from engine.evaluate_drv import EvaluateEngine

        _logger.debug("Testing MSE loss calculation in EvaluateEngine.")
        cfg = ExampleConfig()
        engine = EvaluateEngine(cfg)

        # Create dummy data
        targets = {"preprocess_inputs": torch.tensor([[[0.0, 0.0, 2.0, 4.0, 3.0]]]).float().cuda()}
        logits = {
            "log_p": torch.tensor([[[1.0, 2.0, 2.0]]]).float().cuda(),
        }

        # Calculate MSE loss
        loss_dict = engine.calculate_score(logits, targets)
        self.assertIn("score", loss_dict)
        # For y, it will remove first two dimensions so only [2.0,4.0,3.0] remains
        # out_1 = (1.0-2.0)^2 = 1.0
        # out_2 = (2.0-4.0)^2 = 4.0
        # out_3 = (2.0-3.0)^2 = 1.0
        # total mse_loss = (1.0+4.0+1.0) / 3 = 6.0
        self.assertAlmostEqual(loss_dict["score"], 2.0, places=4)

    def test_load_checkpoint_no_file(self):
        """Test loading checkpoint when no file exists."""
        import tempfile

        from configs.example import Config
        from engine.evaluate_drv import EvaluateEngine

        _logger.debug("Testing loading checkpoint with no existing file in EvaluateEngine.")
        cfg = Config()
        cfg.ckpt_dir = tempfile.mkdtemp()
        cfg.current_time = "non_existent_time"

        engine = EvaluateEngine(cfg)

        model = engine.build_model()
        with self.assertRaises(FileNotFoundError):
            engine.load_checkpoint(model)

    def test_load_checkpoint_with_file(self):
        """Test loading checkpoint when file exists."""
        import os
        import tempfile

        import torch

        from configs.example import Config
        from engine.evaluate_drv import EvaluateEngine

        _logger.debug("Testing loading checkpoint with existing file in EvaluateEngine.")
        cfg = Config()
        cfg.ckpt_dir = tempfile.mkdtemp()
        cfg.current_time = "test_time"

        # Create a dummy model and save a checkpoint
        engine = EvaluateEngine(cfg)
        model = engine.build_model()
        os.makedirs(os.path.join(cfg.ckpt_dir, cfg.current_time), exist_ok=True)
        ckpt_path = os.path.join(cfg.ckpt_dir, cfg.current_time, "latest.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Now load the checkpoint
        engine.load_checkpoint(model)  # Should not raise an error


class TestTrainSTFNetEngine(unittest.TestCase):
    """Test cases for TrainSTFNetEngine."""

    def test_calculate_loss(self):
        """Test STFNet specific method in TrainSTFNetEngine."""
        from configs.STFNet import Config as ExampleConfig
        from engine.train_stfnet import TrainEngine as TrainSTFNetEngine

        _logger.debug("Testing STFNet specific method in TrainSTFNetEngine.")
        cfg = ExampleConfig()
        engine = TrainSTFNetEngine(cfg)

        predictions = {
            "cls_logits": torch.tensor([0.2, -1.5, 0.7]).float().cuda(),
            "mile_logits": torch.tensor([0.5, 0.8, 0.3]).float().cuda(),
            "reg_logits": torch.tensor([[3.0, 4.0]]).float().cuda(),
            "time_series_data": torch.tensor([[[2.5, 3.5]]]).float().cuda(),
        }
        targets_dict = {
            "label": torch.tensor([1, 0, 1]).float().cuda(),
            "norm_mileage": torch.tensor([0.6, 0.9, 0.4]).float().cuda(),
        }

        loss_dict = engine.calculate_loss(predictions, targets_dict)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("bcel_loss", loss_dict)
        self.assertIn("reg_loss", loss_dict)
        self.assertIn("mile_loss", loss_dict)

        # hand-calculated losses for verification
        yn = targets_dict["label"]
        xn = predictions["cls_logits"]
        bcel_loss = -(yn * torch.log(torch.sigmoid(xn)) + (1 - yn) * torch.log(1 - torch.sigmoid(xn)))
        bcel_loss = bcel_loss.mean().item()
        self.assertAlmostEqual(loss_dict["bcel_loss"].item(), bcel_loss, places=4)

        inverted_labels = 1 - yn
        norm_mileage = targets_dict["norm_mileage"]
        mile_logits = predictions["mile_logits"]
        mile_loss = (mile_logits - norm_mileage) ** 2
        mile_loss = (mile_loss * inverted_labels).mean().item()
        self.assertAlmostEqual(loss_dict["mile_loss"].item(), mile_loss, places=4)

        # inverted_labels = 1 - labels
        reg_logits = predictions["reg_logits"]
        time_series_data = predictions["time_series_data"]

        reg_loss = (0.5 * (reg_logits - time_series_data) ** 2).mean(dim=[1, 2])
        reg_loss = (reg_loss * inverted_labels).mean().item()
        self.assertAlmostEqual(loss_dict["reg_loss"].item(), reg_loss, places=4)

        predictions = {
            "cls_logits": torch.tensor([0.2, -1.5, 0.7]).float().cuda(),
            "mile_logits": torch.tensor([0.5, 0.8, 0.3]).float().cuda(),
            "reg_logits": torch.tensor([[1.0, 4.0]]).float().cuda(),
            "time_series_data": torch.tensor([[[3.5, 1.5]]]).float().cuda(),
        }
        targets_dict = {
            "label": torch.tensor([1, 0, 1]).float().cuda(),
            "norm_mileage": torch.tensor([0.6, 0.9, 0.4]).float().cuda(),
        }
        loss_dict = engine.calculate_loss(predictions, targets_dict)
        inverted_labels = 1 - targets_dict["label"]
        reg_logits = predictions["reg_logits"]
        time_series_data = predictions["time_series_data"]

        reg_loss = (torch.abs(reg_logits - time_series_data) - 0.5).mean(dim=[1, 2])
        reg_loss = (reg_loss * inverted_labels).mean().item()
        self.assertAlmostEqual(loss_dict["reg_loss"].item(), reg_loss, places=4)

    def test_calculate_score(self):
        """Test STFNet specific method in TrainSTFNetEngine."""
        from configs.STFNet import Config as ExampleConfig
        from engine.train_stfnet import TrainEngine as TrainSTFNetEngine

        _logger.debug("Testing STFNet specific method in TrainSTFNetEngine.")
        cfg = ExampleConfig()
        engine = TrainSTFNetEngine(cfg)
        predictions = {
            "cls_logits": torch.tensor([0.7]).float().cuda(),
            "mile_logits": torch.tensor([0.5]).float().cuda(),
            "reg_logits": torch.tensor([[3.0]]).float().cuda(),
            "time_series_data": torch.tensor([[[2.5]]]).float().cuda(),
        }
        score = engine.calculate_score(predictions, targets_dict={})
        self.assertIn("score", score)
        expected_score = predictions["cls_logits"].sigmoid().item()
        self.assertAlmostEqual(score["score"], expected_score, places=4)
