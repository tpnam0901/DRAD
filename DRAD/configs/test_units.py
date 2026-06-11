import json
import logging
import os
import shutil
import tempfile
import unittest

from configs.base import Config, import_config

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestConfig(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()

    def test_initially_variable_existence(self):
        """Test that config has self.name attribute initially."""
        _logger.debug("Testing initial existence of 'name' attribute.")
        self.assertTrue(hasattr(self.config, "name"))

    def test_lock_allows_modifying_existing_attributes(self):
        """Test that locked config allows modifying existing attributes."""
        _logger.debug("Testing that locked Config allows modifying existing attributes.")
        original_name = self.config.name
        self.config.name = "modified_name"
        self.assertEqual(self.config.name, "modified_name")
        self.assertNotEqual(self.config.name, original_name)

    def test_unlock_then_lock(self):
        """Test unlocking and then locking again."""
        _logger.debug("Testing unlock then lock cycle.")
        # Unlock to add attribute
        self.config.unlock()
        self.assertFalse(self.config._locked)
        self.config.new_attr = "test_value"
        self.assertEqual(self.config.new_attr, "test_value")

        # Lock again
        self.config.lock()
        self.assertTrue(self.config._locked)

        # Should not be able to add new attributes
        with self.assertRaises(AttributeError):
            self.config.another_attr = "another_value"

        # But can still modify existing ones
        self.config.new_attr = "modified_value"
        self.assertEqual(self.config.new_attr, "modified_value")

    def test_add_new_attribute_when_locked(self):
        """Test that add method fails when config is locked."""
        _logger.debug("Testing add method on locked Config.")
        with self.assertRaises(AttributeError) as context:
            self.config.test_attr = "test_value"

        self.assertIn("Cannot add new attribute 'test_attr'", str(context.exception))
        self.assertIn("locked config", str(context.exception))

    def test_add_new_attribute_when_unlocked(self):
        """Test that add method works when config is unlocked."""
        _logger.debug("Testing add method on unlocked Config.")
        self.config.unlock()
        self.config.test_attr = "test_value"
        self.assertEqual(self.config.test_attr, "test_value")

        # Lock again and ensure attribute exists
        self.config.lock()
        self.assertEqual(self.config.test_attr, "test_value")

    def test_save_config_to_file(self):
        """Test saving config to a JSON file."""
        _logger.debug("Testing save config functionality.")
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "test_config.json")

            # Modify config values
            self.config.name = "test_name"

            # Save config
            self.config.save(config_path)

            # Verify file exists
            self.assertTrue(os.path.exists(config_path))

            # Verify content
            with open(config_path, "r") as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["name"], "test_name")
            self.assertIn("_locked", saved_data)
        finally:
            shutil.rmtree(temp_dir)

    def test_load_config_from_file(self):
        """Test loading config from a JSON file."""
        _logger.debug("Testing load config functionality.")
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "test_config.json")

            # Create test config file
            test_data = {"name": "loaded_name", "_locked": True}
            with open(config_path, "w") as f:
                json.dump(test_data, f)

            # Load config
            self.config.load(config_path)

            # Verify loaded values
            self.assertEqual(self.config.name, "loaded_name")
            self.assertTrue(self.config._locked)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_and_load_cycle(self):
        """Test saving and then loading config maintains values."""
        _logger.debug("Testing save and load cycle.")
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "cycle_config.json")

            # Modify original config
            self.config.name = "cycle_test"
            self.config.unlock()
            self.config.custom_attr = "custom_value"
            self.config.number_attr = 42
            self.config.lock()

            # Save config
            self.config.save(config_path)

            # Create new config instance and load
            new_config = Config()
            new_config.load(config_path)

            # Verify all values match
            self.assertEqual(new_config.name, "cycle_test")
            self.assertEqual(new_config.custom_attr, "custom_value")
            self.assertEqual(new_config.number_attr, 42)
            self.assertTrue(new_config._locked)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_creates_directory(self):
        """Test that save creates nested directories if they don't exist."""
        _logger.debug("Testing save creates nested directories.")
        temp_dir = tempfile.mkdtemp()
        try:
            # Create path with nested directories that don't exist
            config_path = os.path.join(temp_dir, "nested", "dir", "config.json")

            self.config.name = "nested_test"
            self.config.save(config_path)

            # Verify file and directories were created
            self.assertTrue(os.path.exists(config_path))

            # Verify content
            with open(config_path, "r") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["name"], "nested_test")
        finally:
            shutil.rmtree(temp_dir)

    def test_save_with_complex_types(self):
        """Test saving config with lists and dicts."""
        _logger.debug("Testing save with complex data types.")
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "complex_config.json")

            # Add complex attributes
            self.config.unlock()
            self.config.list_attr = [1, 2, 3, "test"]
            self.config.dict_attr = {"key1": "value1", "key2": 100}
            self.config.lock()

            # Save and load
            self.config.save(config_path)

            new_config = Config()
            new_config.load(config_path)

            # Verify complex types
            self.assertEqual(new_config.list_attr, [1, 2, 3, "test"])
            self.assertEqual(new_config.dict_attr, {"key1": "value1", "key2": 100})
        finally:
            shutil.rmtree(temp_dir)

    def test_import_config_function(self):
        """Test the import_config function."""
        _logger.debug("Testing import_config function.")
        # Create temporary config file
        config_path = "configs/base.py"
        cfg: Config = import_config(config_path)
        # assert name with "default" exists
        self.assertTrue(hasattr(cfg, "name"))
        self.assertEqual(cfg.name, "default")


if os.path.exists("configs/example.py"):

    class TestExampleConfig(unittest.TestCase):
        """Test cases for ExampleConfig class."""

        _logger.debug("------------ Setting up TestExampleConfig ------------")

        def setUp(self):
            """Set up test fixtures."""
            from configs.example import Config as ExampleConfig

            self.config = ExampleConfig()

        def test_input_size_exists(self):
            """Test that input_size exists in ExampleConfig."""
            _logger.debug("Testing existence of 'input_size' in ExampleConfig.")
            self.assertTrue(hasattr(self.config, "input_size"))
            self.assertEqual(self.config.input_size, 1)

        def test_name_modified(self):
            """Test that name is modified in ExampleConfig."""
            _logger.debug("Testing modified 'name' in ExampleConfig.")
            self.assertEqual(self.config.name, "example_config")
