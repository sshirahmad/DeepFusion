from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

try:
    import apex
except:
    print("No APEX!")

import numpy as np
import torch
from det3d.builder import _create_learning_rate_scheduler

# from det3d.datasets.kitti.eval_hooks import KittiDistEvalmAPHook, KittiEvalmAPHookV2
from det3d.core import DistOptimizerHook
from det3d.datasets import DATASETS, build_dataloader
from det3d.solver.fastai_optim import OptimWrapper
from det3d.torchie.trainer import DistSamplerSeedHook, Trainer, obj_from_dict
from det3d.utils.print_utils import metric_to_str
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .env import get_root_logger


def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "image",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points",
            "gt_boxes_cls",
            "yolo_map1",
            "yolo_map2",
            "yolo_map3",
            "target_boxes_grid1",
            "target_boxes_grid2",
            "target_boxes_grid3",
            "classifier_map1",
            "classifier_map2",
            "classifier_map3",
            "grid_map1",
            "grid_map2",
            "grid_map3",
            "grid_map4"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking, dtype=torch.float)
        elif k in [
            "obj_mask",
            "noobj_mask",
            "obj_mask1",
            "obj_mask2",
            "obj_mask3",
            "noobj_mask1",
            "noobj_mask2",
            "noobj_mask3"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking, dtype=torch.bool)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_losses(losses):
    log_vars = OrderedDict()
    loss = losses["loss"]  # sum of batches loss
    for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.detach().cpu()

    return loss, log_vars


def parse_second_losses(losses):

    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):

    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True, **kwargs)
        loss, log_vars = parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False, **kwargs)


def batch_processor_ensemble(model1, model2, data, train_mode, **kwargs):
    assert 0, 'deprecated'
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    assert train_mode is False 

    example = example_to_device(data, device, non_blocking=False)
    del data

    preds_dicts1 = model1.pred_hm(example)
    preds_dicts2 = model2.pred_hm(example)
    
    num_task = len(preds_dicts1)

    merge_list = []

    # take the average
    for task_id in range(num_task):
        preds_dict1 = preds_dicts1[task_id]
        preds_dict2 = preds_dicts2[task_id]

        for key in preds_dict1.keys():
            preds_dict1[key] = (preds_dict1[key] + preds_dict2[key]) / 2

        merge_list.append(preds_dict1)

    # now get the final prediciton 
    return model1.pred_result(example, merge_list)


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]


def build_one_cycle_optimizer(model, optimizer_config):
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_config.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,   # TODO: CHECKING LR HERE !!!
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=optimizer_config.bn_wd,
    )

    return optimizer


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


def train_detector(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, img_size=cfg.img_dim, num_gpus=cfg.gpus, dist=distributed,
            shuffle=not ds.test_mode, pin_memory=not ds.test_mode, multiscale=not ds.test_mode and cfg.multiscale
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

        # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, batch_processor, optimizer, lr_scheduler, cfg.test_cfg.img_dim, cfg.test_cfg.iou_thres,
        cfg.work_dir, cfg.log_level
    )

    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if distributed:
        trainer.register_hook(DistSamplerSeedHook())

    # # register eval hooks
    # if validate:
    #     val_dataset_cfg = cfg.data.val
    #     eval_cfg = cfg.get('evaluation', {})
    #     dataset_type = DATASETS.get(val_dataset_cfg.type)
    #     trainer.register_hook(
    #         KittiEvalmAPHookV2(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)


    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, local_rank=cfg.local_rank, contain_pcd=cfg.contain_pcd)
