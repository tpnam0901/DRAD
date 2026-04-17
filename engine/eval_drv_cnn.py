import networks

from .eval_drv_lstm import EvaluateEngine as EvaluateEngineBase


class EvaluateEngine(EvaluateEngineBase):
    def build_model(self):
        """Build the model for training."""
        return networks.CNN(self.cfg)
