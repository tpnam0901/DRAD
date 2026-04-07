import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

import networks
from configs.TransGAN import Config
from data.dataset import build_dataset

from .train_base import TrainEngine as BaseTrainEngine


class TrainEngine(BaseTrainEngine):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__(cfg)
        self.cfg = cfg

    def load_train_dataset(self):
        """Load the training dataset."""
        return build_dataset(
            self.cfg.data_root,
            self.cfg.alpha,
            brand_num=self.cfg.brand_num,
            mode="train",
            removeAbnormal=True,
            logger=self.logger,
        )

    def train_step(self, model_gen, model_dis, batch, optimizer_gen, optimizer_dis):
        """Perform a single training step."""
        if not hasattr(self, "criterion_bcel"):
            self.criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction="mean")
        if not hasattr(self, "criterion_mse"):
            self.criterion_mse = torch.nn.MSELoss(reduction="mean")
        optimizer_gen.zero_grad()
        optimizer_dis.zero_grad()
        batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}

        # Combined update for generator and discriminator
        model_gen.unfreeze()
        model_dis.unfreeze()
        model_gen.train()
        model_dis.train()

        fake_data = model_gen(batch)["logits_rec"]
        real_data = batch["normed_voltage"].unsqueeze(-1)  # B x L x 1
        fake_prediction = model_dis(fake_data)
        real_prediction = model_dis(real_data)

        loss_re = self.criterion_mse(fake_data, real_data)
        loss_adv = self.criterion_bcel(fake_prediction, torch.zeros_like(fake_prediction)) + self.criterion_bcel(
            real_prediction, torch.ones_like(real_prediction)
        )
        lambda_re = loss_re.item() / (loss_re.item() + loss_adv.item() + 1e-8)
        lambda_adv = loss_adv.item() / (loss_re.item() + loss_adv.item() + 1e-8)
        total_loss = lambda_re * loss_re + lambda_adv * loss_adv
        total_loss.backward()
        optimizer_gen.step()
        optimizer_dis.step()

        loss_dict = {"total_loss": total_loss.detach(), "loss_re": loss_re.detach(), "loss_adv": loss_adv.detach()}

        return loss_dict, {"f1": -1}

    def train_epoch(self, model_gen, model_dis, dataloader, optimizer_gen, optimizer_dis, scheduler_gen, scheduler_dis):
        if not hasattr(self, "global_step"):
            self.global_step = 0

        with tqdm(total=len(dataloader), ascii=True) as pbar:
            with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                for batch in dataloader:
                    self.global_step += 1
                    loss_dict, metric_dict = self.train_step(model_gen, model_dis, batch, optimizer_gen, optimizer_dis)
                    postfix = "Epoch {}/{} - ".format(self.global_step // len(dataloader) + 1, self.cfg.num_epochs)
                    postfix += "Total Loss: {:.8f} - ".format(loss_dict["total_loss"].item())
                    postfix += "F1: {:.2f} - ".format(metric_dict["f1"])
                    pbar.set_description(postfix)
                    pbar.update(1)
                    for key, value in loss_dict.items():
                        mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
                    for key, value in metric_dict.items():
                        mlflow.log_metric("train_{}".format(key), value, step=self.global_step)
                scheduler_gen.step()
                scheduler_dis.step()

    def evaluate(self, model, dataloader):
        """Evaluate the model on the given dataloader."""
        model.eval()

        # Initialize loss functions
        if not hasattr(self, "criterion_mse_eval"):
            self.criterion_mse_eval = torch.nn.MSELoss(reduction="none")

        # For each car, calculate the average score across all its samples and use that for evaluation
        car_scores_rec = {}
        car_labels = {}
        for batch in tqdm(dataloader, ascii=True, desc="Evaluating"):
            car_ids = batch["car"].detach().cpu().numpy().tolist()
            labels = batch["label"].detach().cpu().numpy().tolist()
            batch = {key: value.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(batch)
                target = batch["normed_voltage"].unsqueeze(-1)  # B x L x 1
                scores_rec = self.criterion_mse_eval(outputs["logits_rec"], target).mean(dim=[1, 2]).detach().cpu().tolist()

            for car_id, label, score_rec in zip(car_ids, labels, scores_rec):
                if car_id not in car_scores_rec:
                    car_scores_rec[car_id] = []
                car_scores_rec[car_id].append(score_rec)
                car_labels[car_id] = label

        # Average scores for each car
        car_avg_scores_rec = {car_id: np.mean(scores) for car_id, scores in car_scores_rec.items()}

        # Calculate metrics based on average scores
        y_true = [car_labels[car_id] for car_id in car_avg_scores_rec.keys()]
        y_scores_rec = [car_avg_scores_rec[car_id] for car_id in car_avg_scores_rec.keys()]
        metric_dict_rec = self.calculate_metrics(np.array(y_scores_rec), np.array(y_true))

        metric_dict = {}
        for key in metric_dict_rec.keys():
            metric_dict["rec_" + key] = metric_dict_rec[key]

        return metric_dict, None

    def build_model(self):
        """Build the model for training."""
        return getattr(networks, self.cfg.model_type)(self.cfg), getattr(networks, "Discriminator")(self.cfg)

    def run(self):
        """Run the training process."""

        train_dataset = self.load_train_dataset()
        min_mileage, max_mileage = train_dataset.get_min_max_mileage()
        test_dataset = self.load_test_dataset()
        test_dataset.set_min_max_mileage(min_mileage, max_mileage)

        train_dataloader = self.get_dataloader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        test_dataloader = self.get_dataloader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        model_gen, model_dis = self.build_model()
        model_gen.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer_gen = self.build_optimizer(model_gen)
        scheduler_gen = self.build_scheduler(optimizer_gen)

        model_dis.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer_dis = self.build_optimizer(model_dis)
        scheduler_dis = self.build_scheduler(optimizer_dis)

        best_f1_rec = 0.0
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{self.cfg.num_epochs}")
            self.train_epoch(model_gen, model_dis, train_dataloader, optimizer_gen, optimizer_dis, scheduler_gen, scheduler_dis)
            self.save_checkpoint(model_gen, prefix=f"latest")

            with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                mlflow.log_metric("learning_rate", scheduler_gen.get_last_lr()[0], step=self.global_step)
                metric_dict, _ = self.evaluate(model_gen, test_dataloader)
                for key, value in metric_dict.items():
                    self.logger.info(f"Test {key}: {value:.4f}")
                    mlflow.log_metric("test_{}".format(key), value, step=self.global_step)
                if metric_dict["rec_f1"] > best_f1_rec:
                    best_f1_rec = metric_dict["rec_f1"]
                    self.logger.info(f"New best F1 score for reconstruction: {best_f1_rec:.4f}. Saving checkpoint.")
                    self.save_checkpoint(model_gen, prefix="best_rec_f1")

        self.logger.info("Training completed.")
