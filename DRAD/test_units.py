import logging
import unittest

logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.root.level, format="%(name)s - %(levelname)s - %(message)s")

from configs.test_units import *
from data.test_units import *
from engine.test_units import *
from networks.test_units import *
from utils.test_units import *
from data.test_units_naobop import *

if __name__ == "__main__":
    unittest.main()
