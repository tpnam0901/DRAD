from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        # Seed used in the paper: 641, 3136, 3141, 5083, 6019
        self.seed = 641
        self.unlock()
        # Add more configuration parameters as needed

        # Lock the config to prevent further modifications
        self.lock()
        self.model_type = "MachineLearningModel"
        self.name = "ML_brand_{}_{}".format(self.brand_num, self.seed)
