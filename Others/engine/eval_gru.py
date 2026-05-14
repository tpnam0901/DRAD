import logging

from .train_gru import Config, TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TrainEngine")
