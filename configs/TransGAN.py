from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.model_type = "TransGAN"
        self.name = "TransGAN"

        self.unlock()
        # Add more configuration parameters as needed

        self.ckpt_path = "working/checkpoints/RFDBattery/TransGAN_20260403_145620/best_rec_f1.pth"

        # Lock the config to prevent further modifications
        self.lock()
