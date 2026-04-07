from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.num_epochs = 500

        self.brand_num: int = 3

        self.model_type = "DRV"
        self.name = "DRV"

        self.unlock()
        # Add more configuration parameters as needed

        self.rnn_num_layers: int = 1  # Number of rnn layers
        self.rnn_bidirectional: bool = True  # Whether to use bidirectional rnn
        self.rnn_embed_dim: int = 256  # Embedding dimension for fusion layers

        self.input_features = [
            "normed_soc",
            "normed_current",
            # "normed_min_cell_temperature",
            # "normed_max_cell_voltage",
            # "normed_min_cell_voltage",
            # "normed_max_cell_temperature",
            # "normed_voltage",
        ]
        self.output_features = [
            # "normed_soc",
            # "normed_current",
            "normed_min_cell_temperature",
            "normed_max_cell_voltage",
            "normed_min_cell_voltage",
            "normed_max_cell_temperature",
            "normed_voltage",
        ]

        # Lock the config to prevent further modifications
        self.lock()
