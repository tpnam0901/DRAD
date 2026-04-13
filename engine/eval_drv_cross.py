import os
import os.path as osp
from typing import List

import numpy as np
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

        selected_groups = self.select_groups(
            car_normal_ids,
            cars_abnormal_ids,
            num_normal=num_normal,
            num_abnormal=num_abnormal,
            seed=seed,
        )

        return selected_groups

    def load_checkpoint(self, model, prefix: str = "latest"):
        """Load model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        # ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), prefix + ".pth")
        # if not osp.exists(ckpt_path):
        #     ckpt_path = osp.join(
        #         self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.brand_2_name, self.cfg.brand_2_current_time), prefix + ".pth"
        #     )
        ckpt_path = "/home/phuongnam/RelationalEV/working/checkpoints/RFDBattery/DRV_cl_brand3_20260409_160834/best_rec.pth"
        model.load_state_dict(torch.load(ckpt_path))

    def run(self):
        """Run the training process."""

        num_normal = 10
        num_abnormal = 1
        selected_groups = self.build_car_groups(num_normal=num_normal, num_abnormal=num_abnormal, seed=42)

        eval_metrics_dict = {"mean_std": {}, "iqr": {}, "grouping": {}}

        all_group_mean_std_preds = []
        all_group_iqr_preds = []
        all_group_grouping_preds = []
        all_ground_truths = []
        for group_idx, group in enumerate(selected_groups):
            self.logger.info(f"Evaluating group {group_idx}/{len(selected_groups)}: {group}. Loading datasets for each car in the group...")
            each_car_errors = self.eval_group(group)
            gt = [0] * (len(group) - num_abnormal) + [1] * num_abnormal
            all_ground_truths.extend(gt)
            self.plot_error_matrix(each_car_errors, abnormal_car=group[-1])
            self.logger.debug("Outlier detection using mean + 1.5 * std:")
            outliers, car_error_scores = self.find_outlier_with_mean_std(each_car_errors)
            preds = [1 if car_id in outliers else 0 for car_id in group]
            all_group_mean_std_preds.extend(preds)
            self.logger.debug("Ground truth: {}".format(gt))
            self.logger.debug("Predictions: {}".format(preds))
            metric_dict_dis = self.calculate_metrics(np.asarray(gt), np.asarray(preds))
            eval_metrics_dict["mean_std"]["acc"] = eval_metrics_dict["mean_std"].get("acc", []) + [metric_dict_dis["accuracy"]]
            eval_metrics_dict["mean_std"]["f1"] = eval_metrics_dict["mean_std"].get("f1", []) + [metric_dict_dis["f1"]]
            eval_metrics_dict["mean_std"]["precision"] = eval_metrics_dict["mean_std"].get("precision", []) + [metric_dict_dis["precision"]]
            eval_metrics_dict["mean_std"]["recall"] = eval_metrics_dict["mean_std"].get("recall", []) + [metric_dict_dis["recall"]]
            self.logger.debug(
                f"F1: {metric_dict_dis['f1']:.4f}, Precision: {metric_dict_dis['precision']:.4f}, Recall: {metric_dict_dis['recall']:.4f}, Accuracy: {metric_dict_dis['accuracy']:.4f}"
            )
            self.logger.debug("Car error scores: {}".format(car_error_scores))
            self.logger.debug("Outliers: {}, abnormal car in this group: {}".format(outliers.keys(), group[-num_abnormal:]))
            self.logger.debug("--------------------")
            self.logger.debug("Outlier detection using IQR:")
            outliers_iqr, car_error_scores_iqr = self.find_outlier_with_IQR(each_car_errors)
            preds = [1 if car_id in outliers_iqr else 0 for car_id in group]
            all_group_iqr_preds.extend(preds)
            self.logger.debug("Ground truth: {}".format(gt))
            self.logger.debug("Predictions: {}".format(preds))
            metric_dict_IQR = self.calculate_metrics(np.asarray(gt), np.asarray(preds))
            self.logger.debug("Car error scores (IQR): {}".format(car_error_scores_iqr))
            self.logger.debug("Outliers (IQR): {}, abnormal car in this group: {}".format(outliers_iqr.keys(), group[-num_abnormal:]))
            eval_metrics_dict["iqr"]["acc"] = eval_metrics_dict["iqr"].get("acc", []) + [metric_dict_IQR["accuracy"]]
            eval_metrics_dict["iqr"]["f1"] = eval_metrics_dict["iqr"].get("f1", []) + [metric_dict_IQR["f1"]]
            eval_metrics_dict["iqr"]["precision"] = eval_metrics_dict["iqr"].get("precision", []) + [metric_dict_IQR["precision"]]
            eval_metrics_dict["iqr"]["recall"] = eval_metrics_dict["iqr"].get("recall", []) + [metric_dict_IQR["recall"]]
            self.logger.debug(
                f"F1: {metric_dict_IQR['f1']:.4f}, Precision: {metric_dict_IQR['precision']:.4f}, Recall: {metric_dict_IQR['recall']:.4f}, Accuracy: {metric_dict_IQR['accuracy']:.4f}"
            )
            self.logger.debug("==================")
            self.logger.debug("Outlier detection using Grouping method:")
            groups, car_error_scores_iqr = self.find_outlier_with_group(each_car_errors)
            for idx, grp in enumerate(groups):
                self.logger.debug(f"Group {idx + 1}: {grp.keys()}")
            if len(groups) > 1:
                preds = [1 if car_id in groups[1] else 0 for car_id in group]
            else:
                preds = [0] * len(group)
            all_group_grouping_preds.extend(preds)
            self.logger.debug("Ground truth: {}".format(gt))
            self.logger.debug("Predictions: {}".format(preds))
            metric_dict_grouping = self.calculate_metrics(np.asarray(gt), np.asarray(preds))
            eval_metrics_dict["grouping"]["acc"] = eval_metrics_dict["grouping"].get("acc", []) + [metric_dict_grouping["accuracy"]]
            eval_metrics_dict["grouping"]["f1"] = eval_metrics_dict["grouping"].get("f1", []) + [metric_dict_grouping["f1"]]
            eval_metrics_dict["grouping"]["precision"] = eval_metrics_dict["grouping"].get("precision", []) + [
                metric_dict_grouping["precision"]
            ]
            eval_metrics_dict["grouping"]["recall"] = eval_metrics_dict["grouping"].get("recall", []) + [metric_dict_grouping["recall"]]
            self.logger.debug(
                f"F1: {metric_dict_grouping['f1']:.4f}, Precision: {metric_dict_grouping['precision']:.4f}, Recall: {metric_dict_grouping['recall']:.4f}, Accuracy: {metric_dict_grouping['accuracy']:.4f}"
            )
            self.logger.debug("$$$$$")

        print("Final averaged metrics over all groups:")
        for method, vals in eval_metrics_dict.items():
            avg_acc = np.mean(vals["acc"])
            avg_f1 = np.mean(vals["f1"])
            avg_precision = np.mean(vals["precision"])
            avg_recall = np.mean(vals["recall"])
            print(f"Method: {method} - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

        # Overall metrics
        print("Overall metrics across all groups:")
        print("Mean + 1.5 * std method:")
        global_metric_dict_dis = self.calculate_metrics(np.asarray(all_ground_truths), np.asarray(all_group_mean_std_preds))
        print(
            f"Accuracy: {global_metric_dict_dis['accuracy']:.4f}, F1: {global_metric_dict_dis['f1']:.4f}, Precision: {global_metric_dict_dis['precision']:.4f}, Recall: {global_metric_dict_dis['recall']:.4f}"
        )
        print("IQR method:")
        global_metric_dict_iqr = self.calculate_metrics(np.asarray(all_ground_truths), np.asarray(all_group_iqr_preds))
        print(
            f"Accuracy: {global_metric_dict_iqr['accuracy']:.4f}, F1: {global_metric_dict_iqr['f1']:.4f}, Precision: {global_metric_dict_iqr['precision']:.4f}, Recall: {global_metric_dict_iqr['recall']:.4f}"
        )
        print("Grouping method:")
        global_metric_dict_grouping = self.calculate_metrics(np.asarray(all_ground_truths), np.asarray(all_group_grouping_preds))
        print(
            f"Accuracy: {global_metric_dict_grouping['accuracy']:.4f}, F1: {global_metric_dict_grouping['f1']:.4f}, Precision: {global_metric_dict_grouping['precision']:.4f}, Recall: {global_metric_dict_grouping['recall']:.4f}"
        )
