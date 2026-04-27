import torch

import networks
from engine.train_drv import TrainEngine as BaseEngine


class TrainEngine(BaseEngine):

    def build_model(self):
        """Build the model for training."""
        # self.logger.info("Building the model.")
        # Model building logic would go here
        model_class = getattr(networks, self.cfg.model_type)
        self.logger.info(f"Loadding pretrained model from {self.cfg.pretrained_model_path}")
        model = model_class(self.cfg)
        state_dict = torch.load(self.cfg.pretrained_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.freeze_pretrained_weights()
        return model
