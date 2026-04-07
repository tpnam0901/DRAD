import logging
import os
import os.path as osp
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import metrics
from tqdm.auto import tqdm

from configs.DRV import Config
from data.dataset import build_dataset

from .train_drv import TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TrainEngine")
        self.logger.level = logging.INFO
        self.logger.debug("THIS IS A TEST LOGGING DEBUG MESSAGE. IF YOU SEE THIS, LOGGING WORKS!")

        self.alpha = 1
        self.beta = 2

    def calculate_metrics(self, preds: np.ndarray, targets: np.ndarray, num_linspace: int = 1000) -> Dict:
        """Calculate metrics given predictions and targets."""
        metric_dict = {}
        for threshold in np.linspace(np.min(preds), np.max(preds), num=num_linspace):
            precision = metrics.precision_score(targets, preds >= threshold, average="binary", zero_division=0)
            recall = metrics.recall_score(targets, preds >= threshold, average="binary", zero_division=0)
            f1 = metrics.f1_score(targets, preds >= threshold, average="binary", zero_division=0)
            accuracy = metrics.accuracy_score(targets, preds >= threshold)
            if f1 > metric_dict.get("f1", -1):
                metric_dict["accuracy"] = accuracy
                metric_dict["precision"] = precision
                metric_dict["recall"] = recall
                metric_dict["f1"] = f1
                metric_dict["best_threshold"] = threshold

        return metric_dict

    def export_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray, num_linspace: int = 1000, prefix: str = ""):
        """Export confusion matrix given predictions and targets."""
        best_threshold = None
        best_f1 = -1
        for threshold in np.linspace(np.min(preds), np.max(preds), num=num_linspace):
            f1 = metrics.f1_score(targets, preds >= threshold, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        cm = metrics.confusion_matrix(targets, preds >= best_threshold)
        # Move 1s to the top-left corner and 0s to the bottom-right corner
        cm = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])
        # Plot confusion matrix
        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # Remove colorbar
        plt.gca().collections[0].colorbar.remove()
        # plt.title("Confusion Matrix")
        # Customize x and y ticks to show "Anomalous" and "Normal"
        plt.xticks([0.5, 1.5], ["Anomalous", "Normal"])
        plt.yticks([0.5, 1.5], ["Anomalous", "Normal"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        # Move xticks and xlabel to the top
        plt.gca().xaxis.set_label_position("top")
        plt.gca().xaxis.tick_top()

        # Save confusion matrix figure
        cm_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"confusion_matrix_{prefix}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

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
        save_dir = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time))
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f"cm_acar_{abnormal_car}.png")
        plt.savefig(save_path)

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
        save_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), f"ces_acar_{abnormal_car}.png")
        plt.savefig(save_path)

    def select_groups(self, cars_normal, cars_abnormal, num_normal=-1, num_abnormal=1, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        selected_groups = []

        random.shuffle(cars_abnormal)
        random.shuffle(cars_normal)
        if num_normal == -1:
            num_normal = len(cars_normal) // len(cars_abnormal)
            for i, abnormal_car in enumerate(cars_abnormal):
                start_idx = i * num_normal
                selected_normals = cars_normal[start_idx : start_idx + num_normal]
                group = selected_normals + [abnormal_car]
                selected_groups.append(group)
        else:
            abnormal_indices = list(range(len(cars_abnormal)))
            random.shuffle(abnormal_indices)

            # Repeat normal cars to ensure we have enough normal cars for each group
            normal_indices = []
            for i in range(5):
                indices = list(range(len(cars_normal)))
                random.shuffle(indices)
                normal_indices.extend(indices)

            abnormal_idx = 0
            normal_idx = 0

            selected_abnormal_cars = []
            for _ in range(num_abnormal):
                if abnormal_idx >= len(abnormal_indices):
                    break
                selected_abnormal_cars.append(cars_abnormal[abnormal_indices[abnormal_idx]])
                abnormal_idx += 1
            selected_normals = []
            for _ in range(num_normal):
                if normal_idx >= len(normal_indices):
                    raise ValueError(
                        "Not enough normal cars to select from. Please increase the number of normal cars or reduce num_normal."
                    )
                selected_normals.append(cars_normal[normal_indices[normal_idx]])
                normal_idx += 1
            selected_groups.append(selected_normals + selected_abnormal_cars)

        return selected_groups

    def load_checkpoint(self, model, prefix: str = "latest"):
        """Load model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.checkpoint_dir, "{}_{}".format(self.cfg.name, self.cfg.current_time), prefix + ".pth")
        model.load_state_dict(torch.load(ckpt_path))

    def calculate_score(self, predictions: Dict, targets_dict: Dict):
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets_dict (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """

        # Initialize loss functions
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="none")

        # Reconstruction loss
        logits_rec = predictions["logits_rec"]

        normed_time_series = []
        for feature in self.cfg.output_features:
            normed_time_series.append(targets_dict[feature])
        normed_time_series = torch.stack(normed_time_series, dim=2)
        loss_reg = self.criterion_mse(logits_rec, normed_time_series).mean(dim=[1, 2])

        return loss_reg

    def eval_group(self, group: List[int]) -> Dict:
        all_datasets = {}
        for car_id in group:
            if car_id < 201:
                brand_num = 1
            elif car_id < 401:
                brand_num = 2
            else:
                brand_num = 3

            train_dataset = build_dataset(
                data_root=self.cfg.data_root,
                brand_num=brand_num,
                mode="train",
                car_id=car_id,
                logger=self.logger,
                verbose=False,
            )
            min_mileage, max_mileage = train_dataset.get_min_max_mileage()
            test_dataset = build_dataset(
                data_root=self.cfg.data_root,
                brand_num=brand_num,
                mode="test",
                car_id=car_id,
                logger=self.logger,
                verbose=False,
                train_include=False,
            )
            test_dataset.set_min_max_mileage(min_mileage, max_mileage)
            test_dataloader = self.get_dataloader(
                test_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                drop_last=False,
            )

            all_datasets[car_id] = test_dataloader

        each_car_errors = {}
        for car_id in tqdm(group):
            model = self.build_model()
            self.load_checkpoint(model, prefix=f"{car_id}_latest")

            model.eval()
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for car_data_id, car_dataloader in all_datasets.items():
                total_error = 0.0
                total_data = 0
                with torch.no_grad():
                    for batch in car_dataloader:
                        batch = {
                            key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()
                        }
                        outputs = model(batch)
                        scores = self.calculate_score(outputs, batch)
                        for s in scores:
                            total_error += s
                            total_data += 1
                avg_error = total_error / total_data
                car_dict = each_car_errors.get(car_id, {})
                car_dict[car_data_id] = avg_error
                each_car_errors[car_id] = car_dict

        return each_car_errors

    def run(self):
        """Run the training process."""

        car_normal_ids = []
        cars_abnormal_ids = []
        if self.cfg.brand_num == 3:
            meta_data = pd.read_csv(osp.join(self.cfg.data_root, "battery_brand3", "label", "all_label.csv"))
            # Get unique car ids from meta data
            car_normal_ids = meta_data[meta_data["label"] == 0]["car"].unique().tolist()
            cars_abnormal_ids = meta_data[meta_data["label"] == 1]["car"].unique().tolist()
        else:
            raise NotImplementedError(f"Brand number {self.cfg.brand_num} not implemented for dataset loading.")

        print("Normal cars:", car_normal_ids, "\nAbnormal cars:", cars_abnormal_ids)

        num_normal = -1
        num_abnormal = 1
        selected_groups = self.select_groups(car_normal_ids, cars_abnormal_ids, num_normal=num_normal, num_abnormal=num_abnormal)

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
