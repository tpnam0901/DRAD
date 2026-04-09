from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.brand_num: int = 3
        self.unlock()
        # Add more configuration parameters as needed

        # Lock the config to prevent further modifications
        self.lock()
        self.model_type = "MachineLearningModel"
        self.name = "MachineLearningModel_brand{}".format(self.brand_num)
