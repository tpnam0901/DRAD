import logging
import unittest

from data.basedataset import BaseDataset

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestBaseDataset(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def setUp(self):
        """Set up test fixtures."""
        self.x = [i for i in range(10)]
        self.y = [i % 2 for i in range(10)]
        self.dataset = BaseDataset(self.x, self.y)

    def test_len(self):
        """Test that dataset length is correct."""
        _logger.debug("Testing length of BaseDataset.")
        self.assertEqual(len(self.dataset), 10)

    def test_getitem(self):
        """Test that __getitem__ returns correct data."""
        _logger.debug("Testing __getitem__ of BaseDataset.")
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertEqual(item["x"], self.x[i])
            self.assertEqual(item["y"], self.y[i])
