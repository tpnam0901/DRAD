from configs.base import Config as BaseConfig


class Config(BaseConfig):
    """Example configuration class extending the base Config."""

    def __init__(self):
        super(Config, self).__init__()

        # Modify default parameters
        self.name = "DRVShift_{}_{}".format(self.brand, self.seed)

        self.model_type: str = "DRVShift"

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
        self.pretrained_model_path: str = "checkpoints/CL_DRV_brand1_20260415_141406/all_normal_latest.pth"
        # Lock the config to prevent further modifications
        self.lock()
