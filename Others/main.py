import logging
import warnings

warnings.filterwarnings("ignore")  # Ignore all warnings globally

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.root.level, format="%(name)s - %(levelname)s - %(message)s")

import argparse
import random

import numpy as np
import torch
from configs.base import Config, import_config


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="configs/STFNet.py",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-cfg_ckpt",
        "--config_ckpt",
        type=str,
        default="",
        help="Path to the checkpoint .json file to load configuration from.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["t", "e"],
        required=True,
        help="Engine type to use: TrainEngine (t), EvaluateEngine (e).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["CL", "DRV"],
        default="CL",
        help="Training mode: Central or DRV.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    cfg: Config = import_config(args.config)
    if args.engine == "e":
        assert args.config_ckpt != "", "Checkpoint configuration file must be provided for Evaluate engines."
        cfg.load(args.config_ckpt)

    SEED = cfg.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.mode == "CL":
        if cfg.model_type == "DyAD":
            from engine.eval_dyad import EvaluateEngine
            from engine.train_dyad import TrainEngine
        elif cfg.model_type == "AE":
            from engine.eval_ae import EvaluateEngine
            from engine.train_ae import TrainEngine
        elif cfg.model_type == "LSTM":
            from engine.eval_lstm import EvaluateEngine
            from engine.train_lstm import TrainEngine
        elif cfg.model_type == "MachineLearningModel":
            from engine.run_ml import TrainEngine
            from engine.run_ml import TrainEngine as EvaluateEngine
        elif cfg.model_type == "CNN":
            from engine.eval_cnn import EvaluateEngine
            from engine.train_cnn import TrainEngine
        elif cfg.model_type == "GRU" or cfg.model_type == "DRV":
            from engine.eval_gru import EvaluateEngine
            from engine.train_gru import TrainEngine
        elif cfg.model_type == "TransGAN":
            from engine.eval_transgan import EvaluateEngine
            from engine.train_transgan import TrainEngine
        else:
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented for CL mode.")
    elif args.mode == "DRV":
        if args.engine == "t":
            raise NotImplementedError("Training engine for DRV mode is not implemented.")
        if cfg.model_type == "AE":
            from engine.eval_drv_ae import EvaluateEngine
        elif cfg.model_type == "DyAD":
            from engine.eval_drv_dyad import EvaluateEngine
        elif cfg.model_type == "LSTM":
            from engine.eval_drv_lstm import EvaluateEngine
        elif cfg.model_type == "CNN":
            from engine.eval_drv_cnn import EvaluateEngine
        elif cfg.model_type == "TransGAN":
            from engine.eval_drv_transgan import EvaluateEngine
        elif cfg.model_type == "GRU":
            from engine.eval_drv_gru import EvaluateEngine
        else:
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented for DRV mode.")
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented.")

    if args.engine == "t":
        trainer = TrainEngine(cfg)
        trainer.run()
    elif args.engine == "e":
        evaluator = EvaluateEngine(cfg)
        evaluator.run()
