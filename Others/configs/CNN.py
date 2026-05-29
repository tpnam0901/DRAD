from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        # Seed used in the paper: 980, 2025, 3189, 6315, 8455
        self.brand_num = 3
        self.seed = 980

        self.model_type = "CNN"
        self.name = "CNN_{}_{}".format(self.brand_num, self.seed)

        self.unlock()
        # Add more configuration parameters as needed

        # Lock the config to prevent further modifications
        self.lock()
