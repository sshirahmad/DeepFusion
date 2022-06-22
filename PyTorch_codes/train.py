import argparse
import os
import warnings
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
from det3d.torchie.trainer import load_checkpoint

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import torch

from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    get_root_logger,
    provide_determinism,
    train_detector,
)

parser = argparse.ArgumentParser(description="Train a detector")
parser.add_argument("--config", default="./configs/mobilenetv3_yolov3_inria.py", help="train config file path")
parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use (only applicable to non-distributed training)")
parser.add_argument("--seed", required=False, type=int, default=None, help="random seed")


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        provide_determinism(args.seed)
    else:
        provide_determinism()

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

    if cfg.pretrained is not None:
        try:
            load_checkpoint(model, cfg.pretrained, strict=False)
            print("init weight from {}".format(cfg.pretrained))
        except:
            print("no pretrained model at {}".format(cfg.pretrained))

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text
        )

    train_detector(
        model,
        datasets,
        cfg,
        logger=logger,
    )


if __name__ == "__main__":
    main()
