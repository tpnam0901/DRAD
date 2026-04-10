import os
import os.path as osp
from typing import List

import pandas as pd
import torch

from .eval_drv import EvaluateEngine as BaseEvaluateEngine


class EvaluateEngine(BaseEvaluateEngine):

    def build_car_groups(self, num_normal=-1, num_abnormal=1, seed=42) -> List[List[int]]:
        car_normal_ids = []
        cars_abnormal_ids = []

        meta_data = pd.read_csv(osp.join(self.cfg.data_root, "battery_brand3", "label", "all_label.csv"))
        # Get unique car ids from meta data
        car_normal_ids = meta_data[meta_data["label"] == 0]["car"].unique().tolist()
        cars_abnormal_ids = meta_data[meta_data["label"] == 1]["car"].unique().tolist()

        meta_train = pd.read_csv(os.path.join(self.cfg.data_root, "battery_brand2", "label", "train_label.csv"))
        meta_test = pd.read_csv(os.path.join(self.cfg.data_root, "battery_brand2", "label", "test_label.csv"))
        car_normal_ids += (
            meta_train[meta_train["label"] == 0]["car"].unique().tolist() + meta_test[meta_test["label"] == 0]["car"].unique().tolist()
        )
        cars_abnormal_ids += (
            meta_train[meta_train["label"] == 1]["car"].unique().tolist() + meta_test[meta_test["label"] == 1]["car"].unique().tolist()
        )

        # Remove car id 232 and 230 from dataset.
        car_normal_ids = [car_id for car_id in car_normal_ids if car_id not in [232, 230]]

        print("Normal cars:", car_normal_ids, "\nAbnormal cars:", cars_abnormal_ids)

        selected_groups = self.select_groups(car_normal_ids, cars_abnormal_ids, num_normal=num_normal, num_abnormal=num_abnormal, seed=seed)

        return selected_groups

    def load_checkpoint(self, model, prefix: str = "latest"):
        """Load model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), prefix + ".pth")
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(
                self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.brand_2_name, self.cfg.brand_2_current_time), prefix + ".pth"
            )
        model.load_state_dict(torch.load(ckpt_path))
