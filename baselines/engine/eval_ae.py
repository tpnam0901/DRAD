import logging
import os.path as osp

import torch

from configs.AE import Config

from .train_ae import TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TrainEngine")

    def run(self):
        """Run the training process."""
        self.criterion_mse = torch.nn.MSELoss(reduction="none")
        train_dataset = self.load_train_dataset()
        min_mileage, max_mileage = train_dataset.get_min_max_mileage()
        test_dataset = self.load_test_dataset()
        test_dataset.set_min_max_mileage(min_mileage, max_mileage)

        test_dataloader = self.get_dataloader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        model = self.build_model()
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "best_rec.pth")
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        metric_dict, _ = self.evaluate(model, test_dataloader)
        print("Evaluation results for best_rec.pth:")
        for key, value in metric_dict.items():
            print(f"Test {key}: {value:.4f}")
