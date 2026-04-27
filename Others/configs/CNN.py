from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.model_type = "CNN"
        self.name = "CNN_cl_brand{}".format(self.brand_num)

        self.unlock()
        # Add more configuration parameters as needed

        # Lock the config to prevent further modifications
        self.lock()
