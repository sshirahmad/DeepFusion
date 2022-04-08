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
    set_random_seed,
    train_detector,
)

parser = argparse.ArgumentParser(description="Train a detector")
parser.add_argument("--config", help="train config file path")
parser.add_argument("--work_dir", help="the dir to save logs and models")
parser.add_argument("--resume_from", help="the checkpoint file to resume from")
parser.add_argument(
    "--validate",
    action="store_true",
    help="whether to evaluate the checkpoint during training",
)
parser.add_argument(
    "--gpus",
    type=int,
    default=1,
    help="number of gpus to use " "(only applicable to non-distributed training)",
)
parser.add_argument("--seed", required=False, type=int, default=None, help="random seed")
parser.add_argument(
    "--launcher",
    choices=["none", "pytorch", "slurm", "mpi"],
    default="none",
    help="job launcher",
)
parser.add_argument("--local_rank", required=False, type=int, default=0)
parser.add_argument(
    "--autoscale-lr",
    action="store_true",
    help="automatically scale lr with the number of gpus",
)

def main():
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parser.parse_args()
    args.config = "../configs/mobilenetv3_yolov3_inria.py"
    args.gpus = 1

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # copy important files to backup
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

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
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


if __name__ == "__main__":
    main()
