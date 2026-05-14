from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.model_type = "AE"
        self.name = "AE_{}_{}".format(self.brand_num, self.seed)

        self.unlock()
        # Add more configuration parameters as needed

        self.rnn_input_size: int = 4  # Hidden size for rnn layers
        self.fusion_embed_dim: int = 32  # Embedding dimension for fusion layers

        # Lock the config to prevent further modifications
        self.lock()
