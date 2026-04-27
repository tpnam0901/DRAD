import logging
import math
import os.path as osp
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import networks
from configs.base import Config
from data.naobop_dataset import EvalNaoBopDataset
from engine.evaluate_drv import EvaluateEngine as BaseEvaluateEngine

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


class EvaluateEngine(BaseEvaluateEngine):
    def __init__(self, cfg_b1: Config, cfg_b2: Config, cfg_ref: Config):
        super().__init__(cfg_b1)
        self.cfg_b2 = cfg_b2
        self.cfg_ref = cfg_ref

        self.loss_nll = torch.nn.SmoothL1Loss(reduction="mean")
        self.loss_mse = torch.nn.MSELoss(reduction="mean")
        self.best_val_loss = float("inf")
        self.step = 1
        self.logger = logging.getLogger("EvaluateEngine")

        self.alpha = 1
        self.beta = 2

    def build_model2(self):
        """Build the model for training."""
        # self.logger.info("Building the model.")
        # Model building logic would go here
        model_class = getattr(networks, self.cfg_b2.model_type)
        return model_class(self.cfg_b2)

    def build_ref_model(self):
        """Build the model for training."""
        # self.logger.info("Building the model.")
        # Model building logic would go here
        model_class = getattr(networks, self.cfg_ref.model_type)
        return model_class(self.cfg_ref)

    def _group_cars_by_brand(self, each_car_errors):
        # Group cars by brand based on their ids
        brand2 = []
        brand3 = []
        for key in each_car_errors.keys():
            if key > 200 and key < 400:
                brand2.append(key)
            elif key > 400:
                brand3.append(key)
            else:
                raise ValueError(f"Unexpected car id: {key}")
        return brand2, brand3

    def _compute_error_per_brand(self, each_car_errors, brand2, brand3):
        # per group errors for cars in the same brand
        inner_brand2_error = []
        for car_id in brand2:
            for car_id2, error in each_car_errors[car_id].items():
                if car_id2 in brand2:
                    inner_brand2_error.append(error)
        inner_brand3_error = []
        for car_id in brand3:
            for car_id2, error in each_car_errors[car_id].items():
                if car_id2 in brand3:
                    inner_brand3_error.append(error)

        # per group errors for cross brand cars
        all_brand2_error = []
        for car_id in brand2:
            for car_id2, error in each_car_errors[car_id].items():
                all_brand2_error.append(error)
        all_brand3_error = []
        for car_id in brand3:
            for car_id2, error in each_car_errors[car_id].items():
                all_brand3_error.append(error)
        return np.average(inner_brand2_error), np.average(inner_brand3_error), np.average(all_brand2_error), np.average(all_brand3_error)

    def _compute_brand_ratio(self, ref_error, inner_error):
        return ref_error / inner_error

    def refine_error(self, each_car_errors, each_ref_errors):
        brand2, brand3 = self._group_cars_by_brand(each_car_errors)
        inner_brand2_error, inner_brand3_error, all_brand2_error, all_brand3_error = self._compute_error_per_brand(
            each_car_errors, brand2, brand3
        )
        ref_brand2_error, ref_brand3_error, all_ref_brand2_error, all_ref_brand3_error = self._compute_error_per_brand(
            each_ref_errors, brand2, brand3
        )
        ref_brand2_ratio = self._compute_brand_ratio(ref_brand2_error, inner_brand2_error)
        ref_brand3_ratio = self._compute_brand_ratio(ref_brand3_error, inner_brand3_error)
        ref_cross_brand2_ratio = self._compute_brand_ratio(all_ref_brand2_error, all_brand2_error)
        ref_cross_brand3_ratio = self._compute_brand_ratio(all_ref_brand3_error, all_brand3_error)
        # Move from brand3,2 error to refference error by multiplying corresponding EV with the ratio
        for car_id in each_car_errors.keys():
            # Inner brand error shift
            if car_id in brand2:
                for car_id2 in each_car_errors[car_id].keys():
                    if car_id2 in brand2:
                        each_car_errors[car_id][car_id2] *= ref_brand2_ratio
                    elif car_id2 in brand3:
                        each_car_errors[car_id][car_id2] *= ref_cross_brand2_ratio
            elif car_id in brand3:
                for car_id2 in each_car_errors[car_id].keys():
                    if car_id2 in brand3:
                        each_car_errors[car_id][car_id2] *= ref_brand3_ratio
                    elif car_id2 in brand2:
                        each_car_errors[car_id][car_id2] *= ref_cross_brand3_ratio
        return each_car_errors

    def eval_group(self, group: List[int]) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        self.logger.info(f"Evaluating group: {group}")
        all_datasets = {}
        for car_id in group:
            datasets = EvalNaoBopDataset(
                self.cfg.data_root,
                f"fold_{self.cfg.fold_num}_train.txt",
                self.cfg.max_length,
                car_id=car_id,
            )
            if len(datasets) == 0:
                datasets = EvalNaoBopDataset(
                    self.cfg_b2.data_root,
                    f"fold_{self.cfg_b2.fold_num}_train.txt",
                    self.cfg_b2.max_length,
                    car_id=car_id,
                )
            longest_charge_data = []
            for data in iter(datasets):
                longest_charge_data.extend(data)
            all_datasets[car_id] = longest_charge_data

        each_car_errors = {}
        each_car_ref_errors = {}
        for car_id in tqdm(group):
            if car_id > 200 and car_id < 400:
                model = self.build_model2()
            else:
                model = self.build_model()
            model_ref = self.build_ref_model()

            self.load_checkpoint(model, car_id)
            self.load_ref_checkpoint(model_ref)

            model.eval()
            model.cuda()
            model_ref.eval()
            model_ref.cuda()

            for car_data_id, data in all_datasets.items():
                total_error = 0.0
                total_ref_error = 0.0
                total_data = 0
                with torch.no_grad():
                    for charge_data in data:
                        output = model(charge_data)
                        output_ref = model_ref(charge_data)
                        scores = self.calculate_score(output, charge_data)
                        scores_ref = self.calculate_score(output_ref, charge_data)
                        total_error += scores["score"]
                        total_ref_error += scores_ref["score"]
                        total_data += 1
                avg_error = total_error / total_data
                avg_ref_error = total_ref_error / total_data

                car_dict = each_car_errors.get(car_id, {})
                car_dict[car_data_id] = avg_error
                each_car_errors[car_id] = car_dict

                car_ref_dict = each_car_ref_errors.get(car_id, {})
                car_ref_dict[car_data_id] = avg_ref_error
                each_car_ref_errors[car_id] = car_ref_dict

        return each_car_errors, each_car_ref_errors

    def load_ref_checkpoint(self, model):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(
            self.cfg_ref.ckpt_dir,
            "{}_{}".format(self.cfg_ref.name, self.cfg_ref.current_time),
            f"all_normal_latest.pth",
        )
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))

    def load_checkpoint(self, model, car_id: int):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(
            self.cfg.ckpt_dir,
            "{}_{}".format(self.cfg.name, self.cfg.current_time),
            f"car_{car_id}_latest.pth",
        )
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(
                self.cfg_b2.ckpt_dir,
                "{}_{}".format(self.cfg_b2.name, self.cfg_b2.current_time),
                f"car_{car_id}_latest.pth",
            )
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))

    def select_random_groups(self, cars_normal_g1, cars_normal_g2, cars_abnormal, group_size=3, max_abnormal=1, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        selected_groups = []
        selected_gt = []

        normals_per_group = group_size - max_abnormal
        num_groups = math.ceil(len(cars_abnormal) / max_abnormal)

        random.shuffle(cars_abnormal)
        # Repeat normal cars if not enough to fill all groups
        if len(cars_normal_g1) < num_groups * normals_per_group:
            cars_normal_g1 = cars_normal_g1 * (math.ceil((num_groups * normals_per_group) / len(cars_normal_g1)))
        random.shuffle(cars_normal_g1)
        if len(cars_normal_g2) < num_groups * normals_per_group:
            cars_normal_g2 = cars_normal_g2 * (math.ceil((num_groups * normals_per_group) / len(cars_normal_g2)))
        random.shuffle(cars_normal_g2)

        for i in range(num_groups):
            # Half of the normal cars in the group are from brand 1, half from brand 2
            selected_normals_g1 = cars_normal_g1[i * (normals_per_group // 2) : (i + 1) * (normals_per_group // 2)]
            selected_normals_g2 = cars_normal_g2[i * (normals_per_group // 2) : (i + 1) * (normals_per_group // 2)]
            selected_normals = selected_normals_g1 + selected_normals_g2
            selected_abnormals = cars_abnormal[i * max_abnormal : (i + 1) * max_abnormal]
            group = selected_normals + selected_abnormals
            selected_gt.append([0] * len(selected_normals) + [1] * len(selected_abnormals))
            selected_groups.append(group)

        return selected_groups, selected_gt

    def get_groups(self, cars_normal_g1, cars_normal_g2, cars_abnormal):

        group_size = 11
        max_abnormal = 1

        selected_groups, selected_gt = self.select_random_groups(
            cars_normal_g1,
            cars_normal_g2,
            cars_abnormal,
            group_size=group_size,
            max_abnormal=max_abnormal,
        )

        return selected_groups, selected_gt

    def error_refinement(self, error: Dict[int, Dict[int, float]], ref_error: Dict[int, Dict[int, float]]) -> Dict[int, float]:
        refined_error = {}
        for car_id in error.keys():
            car_error_dict = error[car_id]
            car_ref_error_dict = ref_error.get(car_id, {})
            refined_car_error_dict = {}
            for data_id, err in car_error_dict.items():
                ref_err = car_ref_error_dict.get(data_id, 0.0)
                refined_err = self.alpha * err + self.beta * ref_err
                refined_car_error_dict[data_id] = refined_err
            refined_error[car_id] = refined_car_error_dict
        return refined_error

    def run(self):
        """Run the training process."""
        print("Starting evaluation on {} dataset, fold {}".format(self.cfg.data_root, self.cfg.fold_num))

        label_path = osp.join(self.cfg.data_root, "label", "all_label.csv")
        if not osp.exists(label_path):
            label_path = osp.join(self.cfg.data_root, "fold_label.csv")
        all_car_branda = pd.read_csv(label_path)
        cars_normal_g1 = all_car_branda[all_car_branda["label"] == 0]["car"].unique().tolist()
        cars_abnormal = all_car_branda[all_car_branda["label"] == 1]["car"].unique().tolist()

        label_path = osp.join(self.cfg_b2.data_root, "label", "all_label.csv")
        if not osp.exists(label_path):
            label_path = osp.join(self.cfg_b2.data_root, "fold_label.csv")
        all_car_brandb = pd.read_csv(label_path)
        cars_normal_g2 = all_car_brandb[all_car_brandb["label"] == 0]["car"].unique().tolist()

        print("Normal cars:", cars_normal_g1, "\nNormal cars:", cars_normal_g2, "\nAbnormal cars:", cars_abnormal)

        selected_groups, selected_gt = self.get_groups(cars_normal_g1, cars_normal_g2, cars_abnormal)

        metrics = {"mean_std": {}, "iqr": {}, "grouping": {}}
        all_group_errors = []

        all_group_mean_std_preds = []
        all_group_iqr_preds = []
        all_group_grouping_preds = []
        all_ground_truths = []
        all_car_ids = []
        for i, (group, gt) in enumerate(zip(selected_groups, selected_gt)):
            each_car_errors, each_car_ref_errors = self.eval_group(group)
            each_car_errors = self.refine_error(each_car_errors, each_car_ref_errors)

            all_car_ids.extend(group)
            all_group_errors.append(each_car_errors)
            all_ground_truths.extend(gt)
            self.plot_error_matrix(each_car_errors, abnormal_car=group[-1])
            print("Outlier detection using mean + 1.5 * std:")
            outliers, car_error_scores = self.find_outlier_with_mean_std(each_car_errors)
            preds = [1 if car_id in outliers else 0 for car_id in group]
            all_group_mean_std_preds.extend(preds)
            print("Ground truth:", gt)
            print("Predictions:", preds)
            acc, f1, precision, recall = self.calculate_metrics(gt, preds)
            metrics["mean_std"]["acc"] = metrics["mean_std"].get("acc", []) + [acc]
            metrics["mean_std"]["f1"] = metrics["mean_std"].get("f1", []) + [f1]
            metrics["mean_std"]["precision"] = metrics["mean_std"].get("precision", []) + [precision]
            metrics["mean_std"]["recall"] = metrics["mean_std"].get("recall", []) + [recall]
            print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")
            print("Car error scores:", car_error_scores)
            print("Outliers:", outliers.keys(), "abnormal car in this group:", group[-1])
            print("--------------------")
            print("Outlier detection using IQR:")
            outliers_iqr, car_error_scores_iqr = self.find_outlier_with_IQR(each_car_errors)
            preds = [1 if car_id in outliers_iqr else 0 for car_id in group]
            all_group_iqr_preds.extend(preds)
            print("Ground truth:", gt)
            print("Predictions:", preds)
            acc, f1, precision, recall = self.calculate_metrics(gt, preds)
            print(
                "Outliers (IQR):",
                outliers_iqr.keys(),
                "abnormal car in this group:",
                group[-1],
            )
            metrics["iqr"]["acc"] = metrics["iqr"].get("acc", []) + [acc]
            metrics["iqr"]["f1"] = metrics["iqr"].get("f1", []) + [f1]
            metrics["iqr"]["precision"] = metrics["iqr"].get("precision", []) + [precision]
            metrics["iqr"]["recall"] = metrics["iqr"].get("recall", []) + [recall]
            print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")
            print("==================")
            print("Outlier detection using Grouping method:")
            groups, car_error_scores_iqr = self.find_outlier_with_group(each_car_errors)
            for idx, grp in enumerate(groups):
                print(f"Group {idx + 1}: {grp.keys()}")
            if len(groups) > 1:
                preds = [1 if car_id in groups[1] else 0 for car_id in group]
            else:
                preds = [0] * len(group)
            all_group_grouping_preds.extend(preds)
            print("Ground truth:", gt)
            print("Predictions:", preds)
            acc, f1, precision, recall = self.calculate_metrics(gt, preds)
            metrics["grouping"]["acc"] = metrics["grouping"].get("acc", []) + [acc]
            metrics["grouping"]["f1"] = metrics["grouping"].get("f1", []) + [f1]
            metrics["grouping"]["precision"] = metrics["grouping"].get("precision", []) + [precision]
            metrics["grouping"]["recall"] = metrics["grouping"].get("recall", []) + [recall]
            print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")
            print("$$$$$")
        print("Final averaged metrics over all groups:")
        for method, vals in metrics.items():
            avg_acc = np.mean(vals["acc"])
            avg_f1 = np.mean(vals["f1"])
            avg_precision = np.mean(vals["precision"])
            avg_recall = np.mean(vals["recall"])
            print(f"Method: {method} - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

        # Overall metrics
        print("Overall metrics across all groups:")
        print("Mean + 1.5 * std method:")
        acc, f1, precision, recall = self.calculate_metrics(all_ground_truths, all_group_mean_std_preds)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("IQR method:")
        acc, f1, precision, recall = self.calculate_metrics(all_ground_truths, all_group_iqr_preds)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("Grouping method:")
        acc, f1, precision, recall = self.calculate_metrics(all_ground_truths, all_group_grouping_preds)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Overall metrics after remove duplicate car ids (in case some cars appear in multiple groups)
        print("Overall metrics across all groups after removing duplicate car ids:")
        filtered_gt = {}
        filtered_mean_std_preds = {}
        filtered_iqr_preds = {}
        filtered_grouping_preds = {}
        for car_id, gt, mean_std_pred, iqr_pred, grouping_pred in zip(
            all_car_ids,
            all_ground_truths,
            all_group_mean_std_preds,
            all_group_iqr_preds,
            all_group_grouping_preds,
        ):
            filtered_gt[car_id] = gt
            filtered_mean_std_preds.setdefault(car_id, 0)
            filtered_iqr_preds.setdefault(car_id, 0)
            filtered_grouping_preds.setdefault(car_id, 0)
            if mean_std_pred == 1:
                filtered_mean_std_preds[car_id] = 1
            if iqr_pred == 1:
                filtered_iqr_preds[car_id] = 1
            if grouping_pred == 1:
                filtered_grouping_preds[car_id] = 1
        print("Filtered Ground truth:", filtered_gt.keys())
        print("Filtered Ground truth values:", filtered_gt.values())
        print("Filtered Mean + 1.5 * std predictions:", filtered_mean_std_preds.values())
        print("Filtered IQR predictions:", filtered_iqr_preds.values())
        print("Filtered Grouping predictions:", filtered_grouping_preds.values())
        acc, f1, precision, recall = self.calculate_metrics(list(filtered_gt.values()), list(filtered_mean_std_preds.values()))
        print("Mean + 1.5 * std method:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        acc, f1, precision, recall = self.calculate_metrics(list(filtered_gt.values()), list(filtered_iqr_preds.values()))
        print("IQR method:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        acc, f1, precision, recall = self.calculate_metrics(list(filtered_gt.values()), list(filtered_grouping_preds.values()))
        print("Grouping method:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
