import logging
import os.path as osp

import torch

from configs.DRV import Config

from .train_drv import TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TrainEngine")

    def run(self):
        """Run the training process."""
        self.criterion_mse = torch.nn.MSELoss(reduction="none")
        _, test_dataset, _, _ = self.load_data()

        model = self.build_model()
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "best_rec.pth")
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        metric_dict, _ = self.evaluate(model, test_dataset)
        print("Evaluation results for best_rec.pth:")
        for key, value in metric_dict.items():
            print(f"Test {key}: {value:.4f}")
