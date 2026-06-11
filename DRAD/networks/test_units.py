import logging
import unittest

import torch

from configs.example import Config as ExampleConfig
from networks.simpleNN import SimpleNN

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestSimpleNN(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = ExampleConfig()
        self.input_x = torch.randn(5, 1)  # Batch size of 5, input size of 1
        self.model = SimpleNN(self.cfg)

    def test_forward_output_shape(self):
        """Test that the forward method returns output of correct shape."""
        _logger.debug("Testing forward method output shape.")
        output = self.model(self.input_x)["logits"]
        self.assertEqual(output.shape, (5, self.cfg.output_size))
