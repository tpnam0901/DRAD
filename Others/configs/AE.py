from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        # Seed used in the paper for brand 1 (Dahu): 980, 2025, 3189, 6315, 8455
        # Seed used in the paper for brand 3 (Naobop): 1246, 2086, 4464, 5829, 9796
        self.brand_num = 3
        self.seed = 1246

        self.model_type = "AE"
        self.name = "AE_{}_{}".format(self.brand_num, self.seed)

        self.unlock()
        # Add more configuration parameters as needed

        self.rnn_input_size: int = 7  # Hidden size for rnn layers
        self.fusion_embed_dim: int = 32  # Embedding dimension for fusion layers

        # Lock the config to prevent further modifications
        self.lock()
