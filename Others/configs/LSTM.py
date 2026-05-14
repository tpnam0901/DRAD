from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.optimizer = "adamw"
        self.model_type = "LSTM"
        self.name = "LSTM_{}_{}".format(self.brand_num, self.seed)

        self.unlock()
        # Add more configuration parameters as needed

        self.rnn_input_size: int = 4  # Hidden size for rnn layers
        self.rnn_num_layers: int = 3  # Number of rnn layers
        self.rnn_bidirectional: bool = True  # Whether to use bidirectional rnn

        self.fusion_embed_dim: int = 100  # Embedding dimension for fusion layers

        # Lock the config to prevent further modifications
        self.lock()
