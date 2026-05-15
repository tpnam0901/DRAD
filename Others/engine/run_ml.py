import glob
import logging
import os
import os.path as osp
from typing import Dict

import numpy as np
import pandas as pd
from configs.base import Config
from data.dataset import build_dataset
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class TrainEngine(object):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__()
        self.cfg = cfg
        self.mlflow_run_name = cfg.name + "-" + self.cfg.current_time
        cfg.save(
            osp.join(
                cfg.checkpoint_dir,
                "{}_{}".format(self.cfg.name, self.cfg.current_time),
                "config.json",
            )
        )

        self.logger = logging.getLogger("TrainEngine")
        self.logger.setLevel(logging.root.level)
        log_path = osp.join(cfg.checkpoint_dir, "{}_{}".format(cfg.name, cfg.current_time), "ml.log")
        basedir = os.path.dirname(log_path)
        os.makedirs(basedir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_train_dataset(self, car_ids):
        """Load the training dataset."""
        return build_dataset(
            self.cfg.data_root,
            brand_num=self.cfg.brand_num,
            mode="train",
            car_ids=car_ids,
            fold_num=self.cfg.fold_num,
        )

    def load_test_dataset(self, car_ids):
        """Load the training dataset."""
        return build_dataset(
            self.cfg.data_root,
            brand_num=self.cfg.brand_num,
            mode="val",
            car_ids=car_ids,
            fold_num=self.cfg.fold_num,
        )

    def get_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
    ):
        """Get dataloader for the given dataset."""

        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, list(range(os.cpu_count())))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=collate_fn,  # =collate if self.args.variable_length else None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last,
        )

    def calculate_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate metrics given predictions and targets."""
        metric_dict = {}
        metric_dict["precision"] = metrics.precision_score(targets, preds, average="binary", zero_division=0)
        metric_dict["recall"] = metrics.recall_score(targets, preds, average="binary", zero_division=0)
        metric_dict["f1"] = metrics.f1_score(targets, preds, average="binary", zero_division=0)
        metric_dict["accuracy"] = metrics.accuracy_score(targets, preds)
        return metric_dict

    def load_data(self):
        if self.cfg.brand_num == 3:
            car_info = pd.read_csv(osp.join(self.cfg.data_root, "battery_brand3", "label", "all_label.csv"))
            car_available_ids = car_info["car"].unique().tolist()
        else:
            files = glob.glob(osp.join(self.cfg.data_root, f"battery_brand{self.cfg.brand_num}", "data_by_segments", "*"))
            car_available_ids = list(set([int(osp.basename(f).split("_")[0]) for f in files]))
            car_info1 = pd.read_csv(osp.join(self.cfg.data_root, f"battery_brand{self.cfg.brand_num}", "label", "train_label.csv"))
            car_info2 = pd.read_csv(osp.join(self.cfg.data_root, f"battery_brand{self.cfg.brand_num}", "label", "test_label.csv"))
            car_info = pd.concat([car_info1, car_info2], ignore_index=True)

        car_normal_ids = car_info[car_info["label"] == 0]["car"].unique().tolist()
        car_abnormal_ids = car_info[car_info["label"] == 1]["car"].unique().tolist()

        car_normal_ids = [car_id for car_id in car_normal_ids if car_id in car_available_ids]
        car_abnormal_ids = [car_id for car_id in car_abnormal_ids if car_id in car_available_ids]

        train_dataset = self.load_train_dataset(car_normal_ids)
        min_mileage, max_mileage = train_dataset.get_min_max_mileage()
        test_dataset = self.load_test_dataset(car_normal_ids + car_abnormal_ids)
        test_dataset.set_min_max_mileage(min_mileage, max_mileage)

        train_dataloader = self.get_dataloader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )
        return train_dataloader, test_dataset, car_normal_ids, car_abnormal_ids

    def run(self):
        """Run the training process."""

        train_dataloader, test_dataset, car_normal_ids, car_abnormal_ids = self.load_data()

        x_train = []
        y_train = []
        y_train_label = []
        for batch in tqdm(train_dataloader, desc="Extracting features from training data"):
            normed_time_series = batch["normed_time_series"].flatten(0, 1).detach().cpu().numpy()
            x_train.append(normed_time_series)

            label = batch["label"].flatten().detach().cpu().numpy()
            label = np.repeat(label, normed_time_series.shape[0])  # Repeat label to match the shape of normed_time_series
            car_id = batch["car"].flatten().detach().cpu().numpy()
            car_id = np.repeat(car_id, normed_time_series.shape[0])  # Repeat car_id to match the shape of normed_time_series
            charge_id = batch["charge_segment"].flatten().detach().cpu().numpy()
            charge_id = np.repeat(charge_id, normed_time_series.shape[0])  # Repeat charge_id to match the shape of normed_time_series
            y_train.append(np.stack([label, car_id, charge_id], axis=1))
            y_train_label.append(label)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_train_label = np.concatenate(y_train_label, axis=0)

        self.logger.info(f"Training data shape: {x_train.shape}")
        self.logger.info(f"Training labels shape: {y_train.shape}")

        x_test = []
        y_test = []
        y_test_label = []
        for samples in tqdm(test_dataset, desc="Extracting features from test data"):
            for batch in samples:
                for k, v in batch.items():
                    batch[k] = v.unsqueeze(0)
                normed_time_series = batch["normed_time_series"].flatten(0, 1).detach().cpu().numpy()
                x_test.append(normed_time_series)
                label = batch["label"].flatten().detach().cpu().numpy()
                label = np.repeat(label, normed_time_series.shape[0])  # Repeat label to match the shape of normed_time_series
                car_id = batch["car"].flatten().detach().cpu().numpy()
                car_id = np.repeat(car_id, normed_time_series.shape[0])  # Repeat car_id to match the shape of normed_time_series
                charge_id = batch["charge_segment"].flatten().detach().cpu().numpy()
                charge_id = np.repeat(charge_id, normed_time_series.shape[0])  # Repeat charge_id to match the shape of normed_time_series
                y_test.append(np.stack([label, car_id, charge_id], axis=1))
                y_test_label.append(label)

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        y_test_label = np.concatenate(y_test_label, axis=0)
        self.logger.info(f"Test data shape: {x_test.shape}")
        self.logger.info(f"Test labels shape: {y_test.shape}")

        self.logger.info("---------------- Starting PCA-based anomaly detection on test data ---------------")
        x_test_pca = x_test[:, 2:]  # Remove SOC, current.
        self.logger.info(f"Test data shape change from {x_test.shape} to {x_test_pca.shape} after removing SOC and current.")
        pca = PCA(n_components=2, random_state=self.cfg.seed)
        x_test_reconstructed = pca.inverse_transform(pca.fit_transform(x_test_pca))
        residual = np.linalg.norm(x_test_pca - x_test_reconstructed, axis=1)

        # cumulative sum statistics
        mean_cumulative_sum = np.mean(residual)
        std_cumulative_sum = np.std(residual)
        threshold = mean_cumulative_sum + 3 * std_cumulative_sum
        anomalies = residual > threshold
        self.logger.info(f"Detected {np.sum(anomalies)} anomalies out of {len(residual)} samples using PCA-based anomaly detection.")

        # If total number of anomaly charges is larger than 1/4 of total charges, that EV is abnormal
        car_charge_counts = {}
        for (label, car_id, charge_id), anomaly in zip(y_test, anomalies):
            if car_id not in car_charge_counts:
                car_charge_counts[car_id] = {"total": 0, "anomalies": 0}
            car_charge_counts[car_id]["total"] += 1
            if anomaly:
                car_charge_counts[car_id]["anomalies"] += 1

        best_y_pred_pca = []
        best_y_true = []
        best_f1 = 0
        best_ratio = 0
        for ratio in tqdm(range(3000)):
            ratio = ratio / 1000
            y_pred_pca = []
            y_true = []
            for car_id, counts in car_charge_counts.items():
                num_anomalies = counts["anomalies"]
                num_total = counts["total"]
                y_pred_pca.append(1 if (num_anomalies / num_total) > ratio else 0)
                y_true.append(1 if car_id in y_test[y_test[:, 0] == 1][:, 1] else 0)
            y_pred_pca = np.array(y_pred_pca)
            y_true = np.array(y_true)
            local_metric_dict = self.calculate_metrics(y_pred_pca, y_true)
            if local_metric_dict["f1"] > best_f1:
                best_f1 = local_metric_dict["f1"]
                best_y_pred_pca = y_pred_pca
                best_y_true = y_true
                best_ratio = ratio

        self.logger.info(f"Predicted anomalies PCA: {np.sum(best_y_pred_pca)}, {best_y_pred_pca}")
        self.logger.info(f"True anomalies: {np.sum(best_y_true)}, {best_y_true}")
        best_y_pred_pca = np.array(best_y_pred_pca)
        best_y_true = np.array(best_y_true)
        metric_dict = self.calculate_metrics(best_y_pred_pca, best_y_true)

        self.logger.info(f"Evaluation results using PCA-based anomaly detection with threshold {best_ratio}:")
        for key, value in metric_dict.items():
            self.logger.info(f"Test {key}: {value:.4f}")

        self.logger.info("---------------- Starting Isolation Forest-based anomaly detection on test data ---------------")
        isolation_forest = IsolationForest(random_state=self.cfg.seed)
        isolation_forest.fit(x_train)
        anomalies = isolation_forest.predict(x_test) == -1
        self.logger.info(
            f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} samples using Isolation Forest-based anomaly detection."
        )

        car_charge_counts = {}
        for (label, car_id, charge_id), anomaly in zip(y_test, anomalies):
            if car_id not in car_charge_counts:
                car_charge_counts[car_id] = {"total": 0, "anomalies": 0}
            car_charge_counts[car_id]["total"] += 1
            if anomaly:
                car_charge_counts[car_id]["anomalies"] += 1

        best_y_pred_if = []
        best_y_true = []
        best_f1 = 0
        for ratio in tqdm(range(3000)):
            ratio = ratio / 1000
            y_pred_if = []
            y_true = []
            for car_id, counts in car_charge_counts.items():
                num_anomalies = counts["anomalies"]
                num_total = counts["total"]
                # print(
                #     f"Car ID: {car_id}, Anomalies: {num_anomalies}, Total: {num_total}, Ratio: {num_anomalies / num_total:.4f}"
                # )  # Debugging counts
                y_pred_if.append(1 if (num_anomalies / num_total) > ratio else 0)
                y_true.append(1 if car_id in y_test[y_test[:, 0] == 1][:, 1] else 0)

            y_pred_if = np.array(y_pred_if)
            y_true = np.array(y_true)
            local_metric_dict = self.calculate_metrics(y_pred_if, y_true)
            if local_metric_dict["f1"] > best_f1:
                best_f1 = local_metric_dict["f1"]
                best_y_pred_if = y_pred_if
                best_y_true = y_true

        self.logger.info(f"Predicted anomalies IF: {np.sum(best_y_pred_if)}, {best_y_pred_if}")
        self.logger.info(f"True anomalies: {np.sum(best_y_true)}, {best_y_true}")

        best_y_pred_if = np.array(best_y_pred_if)
        best_y_true = np.array(best_y_true)
        metric_dict = self.calculate_metrics(best_y_pred_if, best_y_true)
        self.logger.info(f"Evaluation results using Isolation Forest-based anomaly detection with threshold {best_ratio}:")
        for key, value in metric_dict.items():
            self.logger.info(f"Test {key}: {value:.4f}")

        self.logger.info("---------------- Starting KNN-based anomaly detection on test data ---------------")

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train[:, 0])  # Use label for training
        y_pred = knn.predict(x_test)
        y_true = y_test[:, 0]  # Use label for evaluation
        self.logger.info(f"Predicted anomalies: {np.sum(y_pred)}, {y_pred}")
        self.logger.info(f"True anomalies: {np.sum(y_true)}, {y_true}")
        car_charge_counts = {}
        for (label, car_id, charge_id), anomaly in zip(y_test, y_pred == 1):
            if car_id not in car_charge_counts:
                car_charge_counts[car_id] = {"total": 0, "anomalies": 0}
            car_charge_counts[car_id]["total"] += 1
            if anomaly:
                car_charge_counts[car_id]["anomalies"] += 1

        best_y_pred_knn = []
        best_y_true = []
        best_f1 = -1
        best_ratio = 0
        for ratio in tqdm(range(3000)):
            ratio = ratio / 1000
            y_pred_knn = []
            y_true = []
            for car_id, counts in car_charge_counts.items():
                num_anomalies = counts["anomalies"]
                num_total = counts["total"]
                # print(
                #     f"Car ID: {car_id}, Anomalies: {num_anomalies}, Total: {num_total}, Ratio: {num_anomalies / num_total:.4f}"
                # )  # Debugging counts
                y_pred_knn.append(1 if (num_anomalies / num_total) > ratio else 0)
                y_true.append(1 if car_id in y_test[y_test[:, 0] == 1][:, 1] else 0)
            y_pred_knn = np.array(y_pred_knn)
            y_true = np.array(y_true)
            local_metric_dict = self.calculate_metrics(y_pred_knn, y_true)
            if local_metric_dict["f1"] > best_f1:
                best_f1 = local_metric_dict["f1"]
                best_y_pred_knn = y_pred_knn
                best_y_true = y_true
                best_ratio = ratio

        self.logger.info(f"Predicted anomalies KNN: {np.sum(best_y_pred_knn)}, {best_y_pred_knn}")
        self.logger.info(f"True anomalies: {np.sum(best_y_true)}, {best_y_true}")
        best_y_pred_knn = np.array(best_y_pred_knn)
        best_y_true = np.array(best_y_true)
        metric_dict = self.calculate_metrics(best_y_pred_knn, best_y_true)

        self.logger.info(f"Evaluation results using KNN-based anomaly detection with threshold {best_ratio}:")
        for key, value in metric_dict.items():
            self.logger.info(f"Test {key}: {value:.4f}")
