import os.path as osp

import torch

from engine.evaluate_drv import EvaluateEngine as BaseEvaluateEngine


class EvaluateEngine(BaseEvaluateEngine):
    def load_checkpoint(self, model, car_id: int):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), "global__1000.pth")
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        # print("Loading checkpoint from:", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
        # self.logger.info(f"Loaded checkpoint from {ckpt_path}")

    def get_groups(self, cars_normal, cars_abnormal):
        selected_groups, selected_gt = self.select_groups(cars_normal, cars_abnormal, seed=self.cfg.seed)
        # group_size = 11
        # max_abnormal = 1
        # selected_groups, selected_gt = self.select_random_groups(
        #     cars_normal, cars_abnormal, group_size=group_size, max_abnormal=max_abnormal, seed=self.cfg.seed
        # )

        return selected_groups, selected_gt
