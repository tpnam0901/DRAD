from configs.DRV import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.unlock()
        # Add more configuration parameters as needed

        self.brand_2_name = ""
        self.brand_2_current_time = ""

        # Lock the config to prevent further modifications
        self.lock()
