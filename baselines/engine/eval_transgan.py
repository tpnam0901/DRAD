import logging
import os.path as osp

import torch

from .train_transgan import Config, TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TrainEngine")

    def run(self):
        """Run the training process."""
        # Initialize loss functions
        if not hasattr(self, "criterion_mse_eval"):
            self.criterion_mse_eval = torch.nn.MSELoss(reduction="none")

        _, test_dataset, _, _ = self.load_data()

        model_gen, _ = self.build_model()
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "best_rec_f1.pth")
        model_gen.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model_gen.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        metric_dict, _ = self.evaluate(model_gen, test_dataset)
        print("Evaluation results for best_rec_f1.pth:")
        for key, value in metric_dict.items():
            print(f"Test {key}: {value:.4f}")
