from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.num_epochs = 100

        self.brand_num: int = 3

        self.model_type = "AE"
        self.name = "AE"

        self.unlock()
        # Add more configuration parameters as needed

        self.rnn_input_size: int = 7  # Hidden size for rnn layers
        self.fusion_embed_dim: int = 32  # Embedding dimension for fusion layers

        # Lock the config to prevent further modifications
        self.lock()
