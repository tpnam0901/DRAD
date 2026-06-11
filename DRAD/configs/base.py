import datetime
import importlib
import json
import logging
import os
import sys
from typing import Any, List

import numpy as np


class BaseConfig(object):
    def __init__(self):
        object.__setattr__(self, "_locked", False)

    def __setattr__(self, name: str, value: Any):
        """Override to prevent adding new attributes when locked.
        Args:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        if not hasattr(self, name) and self._locked:
            raise AttributeError(
                f"Cannot add new attribute '{name}' directly to a locked config. \n"
                f"You can only modify existing attributes. \n"
                f"If you want to add new attributes, unlock the config first."
            )
        object.__setattr__(self, name, value)

    def lock(self):
        """Lock the configuration to prevent adding new attributes but allow modifying existing ones."""
        object.__setattr__(self, "_locked", True)

    def unlock(self):
        """Unlock the configuration to allow adding new attributes and modifying existing ones."""
        object.__setattr__(self, "_locked", False)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, path: str):
        """Save configuration to a JSON file.

        Args:
            path (str): Path to the JSON file (e.g., 'config.json' or 'path/to/config.json')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config_dict = {k: v for k, v in vars(self).items()}

        with open(path, "w") as cfg_file:
            json.dump(config_dict, cfg_file, indent=4)

        logging.info(f"Configuration saved to {path}")

    def load(self, path: str):
        """Load configuration from a JSON file.
        Args:
            path (str): Path to the JSON file (e.g., 'config.json' or 'path/to/config.json')
        """

        with open(path, "r") as f:
            data_dict = json.load(f)
        lock_state = data_dict.pop("_locked", True)
        self.unlock()
        for key, value in data_dict.items():
            setattr(self, key, value)
        self._locked = lock_state

        logging.info(f"Configuration loaded from {path}")


class Config(BaseConfig):
    """Config class that can only modify existing BaseConfig attributes."""

    def __init__(self):
        super().__init__()
        self.name = "default"
        # Set all your default configuration parameters here
        # --------------------------------------------------
        # --------------------------------------------------

        # --------------------------------- Data settings
        self.brand: str = "brand3"
        self.data_root: str = f"data/battery_data/battery_{self.brand}"
        self.fold_num: int = 0
        self.max_length: int = 128
        self.overlap: float = 0.0

        # General settings
        self.seed: int = np.random.randint(0, 10000)
        self.ckpt_dir: str = "checkpoints"
        self.current_time: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.keep_only_latest: bool = True

        self.min_loss: float = 0.005
        self.batch_size: int = 128
        self.num_epochs: int = 1000  # Might be changed to 100 for some experiments
        self.num_workers: int = 0
        self.pin_memory: bool = True

        # --------------------------------- Model settings
        self.model_type: str = "DynamicVAE"

        # DyAD
        self.dyad_rnn_type: str = "gru"
        self.dyad_hidden_size: int = 256
        self.dyad_latent_size: int = 16
        self.dyad_encoder_embedding_size: int = 7
        self.dyad_output_embedding_size: int = 5
        self.dyad_decoder_embedding_size: int = 2
        self.dyad_num_layers: int = 1
        self.dyad_bidirectional: bool = True
        self.dyad_variable_length: bool = False

        # LSTM
        self.lstm_hidden_size: int = 256
        self.lstm_latent_size: int = 16
        self.lstm_num_layers: int = 1
        self.lstm_bidirectional: bool = False

        # STFNet

        self.stfnet_spec_n_fft: int = 127
        self.stfnet_spec_win_length: int = 8

        self.stfnet_rnn_input_size: int = 7  # Hidden size for rnn layers
        self.stfnet_rnn_num_layers: int = 2  # Number of rnn layers
        self.stfnet_rnn_bidirectional: bool = True  # Whether to use bidirectional rnn
        self.stfnet_latent_size: int = 16

        self.stfnet_fusion_embed_dim: int = 64  # Embedding dimension for fusion layers
        self.stfnet_fusion_num_heads: int = 4  # Number of heads for multi-head
        self.stfnet_fourier_fusion_dropout: float = 0.2  # Dropout for Fourier fusion layer

        self.stfnet_loss_alpha: float = 1.0
        self.stfnet_loss_beta: float = 0.05

        # Propose2
        self.propose_block_size: int = 128
        self.propose_vocab_size: int = 7
        self.propose_n_layer: int = 8
        self.propose_n_head: int = 8
        self.propose_n_embd: int = 256
        self.propose_dropout: float = 0.0
        self.propose_bias: bool = True

        # FAAE
        self.faae_d_model: int = 40
        self.faae_att_head: int = 8
        self.faae_res_linear: int = 32
        self.faae_sparsity_threshold: float = 0.0004
        self.faae_memory_size: int = 4000
        self.faae_feature_dim: int = 4096

        # TransGAN
        self.transgan_gen_warm_up: int = 2
        self.transgan_num_encoder_layers: int = 3
        self.transgan_num_decoder_layers: int = 3

        self.transgan_num_attention_heads: int = 8

        self.transgan_input_embedding_dimension: int = 128
        self.transgan_hidden_layers: int = 512
        self.transgan_dropout_attention: float = 0.1
        self.transgan_dropout_pe: float = 0.1
        self.loss_transgan_alpha_mse: float = 0.5
        self.loss_transgan_alpha_adv: float = 0.5

        # --------------------------------- Scheduler & Optimizer settings

        self.sgd_momentum: float = 0.9

        # StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, IdentityScheduler, PolyLR
        self.scheduler: str = "StepLR"

        self.learning_rate: float = 1e-3
        self.weight_decay: float = 3e-05
        self.scheduler_last_epoch: int = -1

        # StepLR
        self.lr_step_size: int = 50
        self.lr_step_gamma: float = 0.5

        # MultiStepLR
        self.lr_milestones: List[int] = [50, 100, 150, 200]
        self.lr_multistep_gamma: float = 0.1

        # ExponentialLR
        self.lr_exp_gamma: float = 0.99

        # CosineAnnealingLR
        self.lr_T_max: int = 50
        self.lr_eta_min: float = 0.00001

        # ReduceLROnPlateau
        self.lr_plateau_mode: str = "min"
        self.lr_plateau_factor: float = 0.1
        self.lr_plateau_patience: int = 10
        self.lr_plateau_threshold: float = 0.0001
        self.lr_plateau_threshold_mode: str = "rel"
        self.lr_plateau_cooldown: int = 0
        self.lr_plateau_min_lr: float = 0
        self.lr_plateau_eps: float = 1e-08

        # CosineAnnealingWarmRestarts
        self.lr_T_0: int = 50
        self.lr_T_mult: int = 2
        self.lr_eta_min: float = 1e-6

        # IdentityScheduler - No params, update every step

        # --------------------------------------------------
        # --------------------------------------------------
        # Lock the configuration to prevent adding new attributes
        object.__setattr__(self, "_locked", True)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
