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
        default="configs/DRV.py",
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
        "-c",
        "--cross",
        action="store_true",
        help="Whether to use the cross-evaluation engine (applicable for EvaluateEngine).",
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

    if cfg.model_type == "DRV":
        from engine.eval_drv import EvaluateEngine

        if args.cross:
            from engine.eval_drv_cross import EvaluateEngine
        else:
            from engine.train_drv import TrainEngine
    elif cfg.model_type == "AE":
        from engine.eval_ae import EvaluateEngine
    elif cfg.model_type == "LSTM":
        from engine.eval_lstm import EvaluateEngine
    elif cfg.model_type == "DyAD":
        from engine.eval_dyad import EvaluateEngine
    elif cfg.model_type == "TransGAN":
        from engine.eval_transgan import EvaluateEngine
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not implemented for CL mode.")

    if args.engine == "t":
        trainer = TrainEngine(cfg)
        trainer.run()
    elif args.engine == "e":
        evaluator = EvaluateEngine(cfg)
        evaluator.run()
