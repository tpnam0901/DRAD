import logging

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
        default="configs/base.py",
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
        "-cfg_ckpt2",
        "--config_ckpt2",
        type=str,
        default="",
        help="Path to the second checkpoint .json file to load configuration from (for cross-evaluation).",
    )

    parser.add_argument(
        "-cfg_ref",
        "--config_ref",
        type=str,
        default="",
        help="Path to the second checkpoint .json file to load configuration from (for cross-evaluation).",
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
        choices=["DRV", "FL", "CL", "DRV_cross", "CL_cross"],
        default="DRV",
        help="Mode to use: DRV (DRV), FL (Federated Learning), CL (Centralized Learning) or DRV_cross (Cross-evaluation for DRV).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    cfg: Config = import_config(args.config)
    cfg.name = f"{args.mode}_{cfg.name}"
    if args.engine == "e":
        assert args.config_ckpt != "", "Checkpoint configuration file must be provided for Evaluate engines."
        cfg.load(args.config_ckpt)
    if args.mode == "DRV_cross":
        assert args.config_ckpt2 != "", "Second checkpoint configuration file must be provided for DRV_cross mode."
        assert args.config_ref != "", "Reference configuration file must be provided for DRV_cross mode."
        assert args.engine == "e", "DRV_cross mode is only applicable for Evaluate engines."
        cfg_b2: Config = import_config(args.config)
        cfg_b2.load(args.config_ckpt2)
        cfg_ref: Config = import_config(args.config)
        cfg_ref.load(args.config_ref)

    SEED = cfg.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.mode == "DRV":
        if cfg.model_type == "DRV":
            from engine.evaluate_drv import EvaluateEngine
            from engine.train_drv import TrainEngine
        elif cfg.model_type == "LSTM":
            from engine.evaluate_drv import EvaluateEngine
            from engine.train_drv import TrainEngine
        elif cfg.model_type == "DRVShift":
            from engine.train_drv_shift import TrainEngine
        else:
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented for DRV mode.")
    elif args.mode == "FL":
        from engine.evaluate_fl import EvaluateEngine
        from engine.train_fl import TrainEngine

    elif args.mode == "CL":
        if cfg.model_type == "DRV":
            from engine.evaluate_cen import EvaluateEngine
            from engine.train_cen import TrainEngine
        else:
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented for CL mode.")

    if args.engine == "t":
        trainer = TrainEngine(cfg)
        trainer.run()
    elif args.engine == "e":
        if args.mode == "DRV_cross":
            from engine.evaluate_drv_cross import EvaluateEngine

            evaluator = EvaluateEngine(cfg, cfg_b2, cfg_ref)
        else:
            evaluator = EvaluateEngine(cfg)
        evaluator.run()
