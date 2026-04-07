from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.model_type = "LSTM"
        self.name = "LSTM"

        self.unlock()
        # Add more configuration parameters as needed

        self.ckpt_path = "working/checkpoints/RFDBattery/LSTM_20260403_145613/best_rec.pth"

        # Lock the config to prevent further modifications
        self.lock()
