from configs.base import Config as BaseConfig


class Config(BaseConfig):
    """Example configuration class extending the base Config."""

    def __init__(self):
        super(Config, self).__init__()
        # Seed used in the paper for brand3 (Naobop): 980, 2025, 3189, 6315, 8455
        self.seed = 8455 
        self.brand = "brand3"
        self.data_root = f"data/battery_data/battery_{self.brand}"
        # Modify default parameters
        self.name = "DRV_{}_{}".format(self.brand, self.seed)

        self.model_type: str = "DRV"
        self.min_loss: float = 0.22
        # DyAD
        self.dyad_rnn_type: str = "gru"
        self.dyad_hidden_size: int = 256
        self.dyad_latent_size: int = 16
        self.dyad_encoder_embedding_size: int = 2  # 2  # 4
        self.dyad_output_embedding_size: int = 5  # 5  # 3
        self.dyad_decoder_embedding_size: int = 5  # 5  # 3
        self.dyad_num_layers: int = 1
        self.dyad_bidirectional: bool = True
        self.dyad_variable_length: bool = False

        self.unlock()
        # Add more configuration parameters as needed

        # Lock the config to prevent further modifications
        self.lock()
