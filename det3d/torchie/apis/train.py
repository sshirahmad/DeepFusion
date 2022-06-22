from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

import numpy as np
import torch
from det3d.builder import _create_learning_rate_scheduler
from det3d.datasets import DATASETS, build_dataloader
from det3d.torchie.trainer import Trainer, obj_from_dict
from torch import nn
from .env import get_root_logger


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    if not paramwise_options:
        group_decay = []
        group_no_decay = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Conv2d):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
        for p in model.named_parameters():
            if p[0] in ['bbox_head.sigma_class', 'bbox_head.sigma_loc', 'bbox_head.sigma_grid']:
                group_no_decay.append(p[1])

        assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(groups, **optimizer_cfg)

    else:
        group_decay = []
        group_no_decay = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Conv2d):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        for p in model.named_parameters():
            if p[0] in ['bbox_head.sigma_class', 'bbox_head.sigma_loc', 'bbox_head.sigma_grid']:
                group_no_decay.append(p[1])

        assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(groups, **optimizer_cfg)


def build_lr_scheduler(optimizer, lr_scheduler_cfg):
    """Build learning rate scheduler from configs.
    Args:
        lr_scheduler_cfg (dict): The config dict of the learning rate scheduler.
            Positional fields are:
                - type: class name of the learning rate scheduler.
            Optional fields are:
                - any arguments of the corresponding learning rate scheduler type, e.g.,
                  milestones, gamma, etc.
    Returns:
        torch.optim.lr_scheduler: The initialized lr scheduler.
    """

    lr_scheduler_cfg = lr_scheduler_cfg.copy()
    lr_scheduler_cfg.optimizer = optimizer

    return obj_from_dict(lr_scheduler_cfg, torch.optim.lr_scheduler)


def train_detector(model, dataset, cfg, logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, num_gpus=cfg.gpus, shuffle=not ds.test_mode
        )
        for ds in dataset
    ]

    total_steps = cfg.total_epochs * len(data_loaders[0])
    print(f"total_steps: {total_steps}")

    if cfg.lr_config.type in ["cosine_warmup", "linear_warmup", "polynomial_warmup", "range_test"]:
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = _create_learning_rate_scheduler(optimizer, cfg.lr_config, total_steps)
        cfg.lr_config = None
    elif cfg.lr_config.type == "OneCycleLR":
        optimizer = build_optimizer(model, cfg.optimizer)
        cfg.lr_config.total_steps = total_steps
        lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_config)
        cfg.lr_config = None
    elif cfg.lr_config.type == "ReduceLROnPlateau":
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_config)
        cfg.lr_config = None
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_config)
        cfg.lr_config = None

    model = model.cuda()
    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, optimizer, lr_scheduler, cfg.test_cfg.img_dim, cfg.test_cfg.iou_thres,
        cfg.work_dir, cfg.log_level
    )

    optimizer_config = cfg.optimizer_config

    # register hooks
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, contain_pcd=cfg.contain_pcd)
