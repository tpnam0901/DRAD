import logging
import math
import os.path as osp
import random
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from configs.base import Config
from data.naobop_dataset import EvalNaoBopDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from engine.train import TrainEngine

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loss_nll = torch.nn.SmoothL1Loss(reduction="mean")
        self.loss_mse = torch.nn.MSELoss(reduction="mean")
        self.best_val_loss = float("inf")
        self.step = 1
        self.logger = logging.getLogger("EvaluateEngine")

        self.alpha = 1
        self.beta = 2

    def calculate_score(self, predictions: Dict, targets_dict: Dict) -> Dict:
        """Calculate MSE loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated MSE loss values.
        """
        logits = predictions["log_p"]
        targets = targets_dict["preprocess_inputs"]
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)
        targets = targets[:, :, self.cfg.dyad_encoder_embedding_size :].float().cuda()
        loss = self.loss_mse(logits, targets)
        return {"score": loss.float().item()}

    def calculate_metrics(
        self,
        labels: Union[List, np.ndarray, pd.Series],
        scores: Union[List, np.ndarray, pd.Series],
    ) -> Tuple[float, float, float, float]:
        """Calculate AUC and best F1 score given labels and scores.
        Args:
            labels (Union[List, np.ndarray]): Ground truth labels.
            scores (Union[List, np.ndarray]): Predicted scores.
        Returns:
            Tuple: AUC score, best F1 score, precision, recall, and threshold.
        """
        acc = float(accuracy_score(labels, scores))
        precision = float(precision_score(labels, scores, zero_division=0))
        recall = float(recall_score(labels, scores, zero_division=0))
        f1 = float(f1_score(labels, scores, zero_division=0))

        return acc, f1, precision, recall

    def eval_group(self, group: List[int]) -> Dict:
        self.logger.info(f"Evaluating group: {group}")
        all_datasets = {}
        for car_id in group:
            datasets = EvalNaoBopDataset(
                self.cfg.data_root,
                f"fold_{self.cfg.fold_num}_train.txt",
                self.cfg.max_length,
                car_id=car_id,
            )
            longest_charge_data = []
            for data in iter(datasets):
                longest_charge_data.extend(data)
            all_datasets[car_id] = longest_charge_data

        each_car_errors = {}
        for car_id in tqdm(group):
            model = self.build_model()
            # try:
            self.load_checkpoint(model, car_id)
            # except:
            #     print(f"Cannot load model for car {car_id}")
            #     continue
            model.eval()
            model.cuda()
            for car_data_id, data in all_datasets.items():
                total_error = 0.0
                total_data = 0
                with torch.no_grad():
                    for charge_data in data:
                        output = model(charge_data)
                        scores = self.calculate_score(output, charge_data)
                        total_error += scores["score"]
                        total_data += 1
                avg_error = total_error / total_data
                car_dict = each_car_errors.get(car_id, {})
                car_dict[car_data_id] = avg_error
                each_car_errors[car_id] = car_dict

        return each_car_errors

    def find_outlier_with_mean_std(self, each_car_errors: Dict) -> Tuple[Dict, Dict]:
        confusion_matrix = np.zeros((len(each_car_errors), len(each_car_errors)))
        for i, car_id in enumerate(each_car_errors.keys()):
            for j, car_data_id in enumerate(each_car_errors.keys()):
                confusion_matrix[i, j] = each_car_errors[car_id][car_data_id]
        for i in range(len(each_car_errors)):
            for j in range(len(each_car_errors)):
                if confusion_matrix[i, j] < confusion_matrix[i, i]:
                    confusion_matrix[i, j] = 0.0
        for i in range(len(each_car_errors)):
            confusion_matrix[i, :] = confusion_matrix[i, :] - confusion_matrix[i, i]
        confusion_matrix[confusion_matrix < 0] = 0.0

        # For each row_i and col_i, find sum of that row and column as the error score of that car
        car_error_scores = {}
        for i, car_id in enumerate(each_car_errors.keys()):
            row_sum = np.sum(confusion_matrix[i, :]) / len(each_car_errors)
            col_sum = np.sum(confusion_matrix[:, i]) / len(each_car_errors)
            car_error_scores[car_id] = self.alpha * row_sum + self.beta * col_sum
        # Find outliers as cars with error score greater than mean + 1.5 * std
        error_values = np.array(list(car_error_scores.values()))
        mean_error = np.mean(error_values)
        std_error = np.std(error_values)
        threshold = mean_error + 1.5 * std_error
        outliers = {car_id: score for car_id, score in car_error_scores.items() if score > threshold}
        return outliers, car_error_scores

    def find_outlier_with_IQR(self, each_car_errors: Dict) -> Tuple[Dict, Dict]:
        confusion_matrix = np.zeros((len(each_car_errors), len(each_car_errors)))
        for i, car_id in enumerate(each_car_errors.keys()):
            for j, car_data_id in enumerate(each_car_errors.keys()):
                confusion_matrix[i, j] = each_car_errors[car_id][car_data_id]
        for i in range(len(each_car_errors)):
            for j in range(len(each_car_errors)):
                if confusion_matrix[i, j] < confusion_matrix[i, i]:
                    confusion_matrix[i, j] = 0.0
        for i in range(len(each_car_errors)):
            confusion_matrix[i, :] = confusion_matrix[i, :] - confusion_matrix[i, i]
        confusion_matrix[confusion_matrix < 0] = 0.0

        # For each row_i and col_i, find sum of that row and column as the error score of that car
        car_error_scores = {}
        for i, car_id in enumerate(each_car_errors.keys()):
            row_sum = np.sum(confusion_matrix[i, :]) / len(each_car_errors)
            col_sum = np.sum(confusion_matrix[:, i]) / len(each_car_errors)
            car_error_scores[car_id] = self.alpha * row_sum + self.beta * col_sum
        # Find outliers using IQR method
        error_values = np.array(list(car_error_scores.values()))
        Q1 = np.percentile(error_values, 25)
        Q3 = np.percentile(error_values, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = {car_id: score for car_id, score in car_error_scores.items() if score > upper_bound}
        return outliers, car_error_scores

    def find_outlier_with_group(self, each_car_errors: Dict) -> Tuple[List[Dict], Dict]:
        confusion_matrix = np.zeros((len(each_car_errors), len(each_car_errors)))
        for i, car_id in enumerate(each_car_errors.keys()):
            for j, car_data_id in enumerate(each_car_errors.keys()):
                confusion_matrix[i, j] = each_car_errors[car_id][car_data_id]
        for i in range(len(each_car_errors)):
            for j in range(len(each_car_errors)):
                if confusion_matrix[i, j] < confusion_matrix[i, i]:
                    confusion_matrix[i, j] = 0.0
        for i in range(len(each_car_errors)):
            confusion_matrix[i, :] = confusion_matrix[i, :] - confusion_matrix[i, i]
        confusion_matrix[confusion_matrix < 0] = 0.0

        # For each row_i and col_i, find sum of that row and column as the error score of that car
        car_error_scores = {}
        for i, car_id in enumerate(each_car_errors.keys()):
            row_sum = np.sum(confusion_matrix[i, :]) / len(each_car_errors)
            col_sum = np.sum(confusion_matrix[:, i]) / len(each_car_errors)

            car_error_scores[car_id] = self.alpha * row_sum + self.beta * col_sum

        # 1. Sort by value
        sorted_items = sorted(car_error_scores.items(), key=lambda x: x[1])
        values = [item[1] for item in sorted_items]

        # 2. Find all gaps between points
        gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        if not gaps:
            return [car_error_scores], car_error_scores

        # 3. Find the biggest gap
        max_gap = max(gaps)
        max_gap_index = gaps.index(max_gap)

        # 4. Threshold Logic: Is this gap "big enough" to be a separate group?
        # We check if the max gap is at least 3x the average of other gaps
        avg_other_gaps = np.mean(gaps)

        if max_gap > (avg_other_gaps * 2):
            # Split into two groups
            group1 = dict(sorted_items[: max_gap_index + 1])
            group2 = dict(sorted_items[max_gap_index + 1 :])
            return [group1, group2], car_error_scores

        else:
            # Only one group exists
            return [car_error_scores], car_error_scores

    def plot_error_matrix(self, each_car_errors: Dict, abnormal_car: int):
        confusion_matrix = np.zeros((len(each_car_errors), len(each_car_errors)))
        for i, car_id in enumerate(each_car_errors.keys()):
            for j, car_data_id in enumerate(each_car_errors.keys()):
                # confusion_matrix[i, j] = each_car_errors[car_id][car_data_id]
                confusion_matrix[j, i] = each_car_errors[car_id][car_data_id]  # swap i and j for correct orientation in the paper
        # If error is lower than diagonal, set to 0
        # for i in range(len(each_car_errors)):
        #     for j in range(len(each_car_errors)):
        #         if confusion_matrix[i, j] < confusion_matrix[i, i]:
        #             confusion_matrix[i, j] = 0.0

        for i in range(len(each_car_errors)):
            confusion_matrix[i, :] = confusion_matrix[i, :] - confusion_matrix[i, i]
        confusion_matrix[confusion_matrix < 0] = 0.0

        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Average MSE Loss")
        plt.xticks(ticks=np.arange(len(each_car_errors)), labels=list(each_car_errors.keys()), rotation=45)
        plt.yticks(ticks=np.arange(len(each_car_errors)), labels=list(each_car_errors.keys()))
        # plt.xlabel("Car Data ID")
        # plt.ylabel("Model Car ID")
        # Swap xlabel and ylabel for correct orientation in the paper
        plt.xlabel("EV Model ID")
        plt.ylabel("EV Data ID")
        plt.title("Error Matrix of Models vs. Data. Abnormal EV is: EV " + str(abnormal_car))
        plt.tight_layout()
        plt.savefig(osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"cm_acar_{abnormal_car}.png"))

        # For each row_i and col_i, find sum of that row and column as the error score of that car
        car_error_scores = {}
        for i, car_id in enumerate(each_car_errors.keys()):
            row_sum = np.sum(confusion_matrix[i, :]) / len(each_car_errors)
            col_sum = np.sum(confusion_matrix[:, i]) / len(each_car_errors)
            # car_error_scores[car_id] = self.alpha * row_sum + self.beta * col_sum
            car_error_scores[car_id] = (
                self.beta * row_sum + self.alpha * col_sum
            )  # swap alpha and beta for correct orientation in the paper
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(car_error_scores)), list(car_error_scores.values()), align="center")
        plt.xticks(ticks=np.arange(len(car_error_scores)), labels=list(car_error_scores.keys()), rotation=45)
        plt.ylabel("Error Score")
        plt.title("Abnormal EV is: EV " + str(abnormal_car))
        plt.tight_layout()
        plt.savefig(osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"ces_acar_{abnormal_car}.png"))

    def load_checkpoint(self, model, car_id: int):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"car_{car_id}_latest.pth")
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        # print("Loading checkpoint from:", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
        # self.logger.info(f"Loaded checkpoint from {ckpt_path}")

    def select_groups(self, cars_normal, cars_abnormal, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        selected_groups = []
        selected_gt = []
        random.shuffle(cars_abnormal)
        random.shuffle(cars_normal)

        num_normal_per_group = len(cars_normal) // len(cars_abnormal)
        for i, abnormal_car in enumerate(cars_abnormal):
            start_idx = i * num_normal_per_group
            selected_normals = cars_normal[start_idx : start_idx + num_normal_per_group]
            group = selected_normals + [abnormal_car]
            selected_groups.append(group)
            selected_gt.append([0] * len(selected_normals) + [1])
        print("The number of members in each group:", [len(group) for group in selected_groups])

        return selected_groups, selected_gt

    def select_random_groups(self, cars_normal, cars_abnormal, group_size=3, max_abnormal=1, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        selected_groups = []
        selected_gt = []

        normals_per_group = group_size - max_abnormal
        num_groups = math.ceil(len(cars_abnormal) / max_abnormal)

        random.shuffle(cars_normal)
        random.shuffle(cars_abnormal)
        # Repeat normal cars if not enough to fill all groups
        if len(cars_normal) < num_groups * normals_per_group:
            cars_normal = cars_normal * (math.ceil((num_groups * normals_per_group) / len(cars_normal)))

        for i in range(num_groups):
            selected_normals = cars_normal[i * normals_per_group : (i + 1) * normals_per_group]
            selected_abnormals = cars_abnormal[i * max_abnormal : (i + 1) * max_abnormal]
            group = selected_normals + selected_abnormals
            selected_gt.append([0] * len(selected_normals) + [1] * len(selected_abnormals))
            selected_groups.append(group)

        return selected_groups, selected_gt

    def get_groups(self, cars_normal, cars_abnormal):
        # selected_groups, selected_gt = self.select_groups(cars_normal, cars_abnormal, seed=2025)

        group_size = 11
        max_abnormal = 1
        selected_groups, selected_gt = self.select_random_groups(
            cars_normal, cars_abnormal, group_size=group_size, max_abnormal=max_abnormal, seed=2025
        )

        return selected_groups, selected_gt

    def run(self):
        """Run the training process."""
        print("Starting evaluation on {} dataset, fold {}".format(self.cfg.data_root, self.cfg.fold_num))

        if self.cfg.brand == "brand3":
            car_info = pd.read_csv(osp.join(self.cfg.data_root, "label", "all_label.csv"))
            car_available_ids = car_info["car"].unique().tolist()
        else:
            with open(osp.join(self.cfg.data_root, f"fold_{self.cfg.fold_num}_train.txt"), "r") as f:
                car_info = f.readlines()
            car_available_ids = list(set([int(osp.basename(f).split("_")[0]) for f in car_info]))
            car_info1 = pd.read_csv(osp.join(self.cfg.data_root, "label", "train_label.csv"))
            car_info2 = pd.read_csv(osp.join(self.cfg.data_root, "label", "test_label.csv"))
            car_info = pd.concat([car_info1, car_info2], ignore_index=True)

        cars_normal = car_info[car_info["label"] == 0]["car"].unique().tolist()
        cars_abnormal = car_info[car_info["label"] == 1]["car"].unique().tolist()

        cars_normal = [car_id for car_id in cars_normal if car_id in car_available_ids]
        cars_abnormal = [car_id for car_id in cars_abnormal if car_id in car_available_ids]

        print("Normal cars:", cars_normal, "\nAbnormal cars:", cars_abnormal)

        selected_groups, selected_gt = self.get_groups(cars_normal, cars_abnormal)

        metrics = {"mean_std": {}, "iqr": {}, "grouping": {}}
        all_group_errors = []

        all_group_mean_std_preds = []
        all_group_iqr_preds = []
        all_group_grouping_preds = []
        all_ground_truths = []
        all_car_ids = []
        for i, (group, gt) in enumerate(zip(selected_groups, selected_gt)):
            each_car_errors = self.eval_group(group)
            all_group_errors.append(each_car_errors)
            all_car_ids.extend(group)
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
            print("Car error scores (IQR):", car_error_scores_iqr)
            print("Outliers (IQR):", outliers_iqr.keys(), "abnormal car in this group:", group[-1])
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
            all_car_ids, all_ground_truths, all_group_mean_std_preds, all_group_iqr_preds, all_group_grouping_preds
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
