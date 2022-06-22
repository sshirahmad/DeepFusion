import itertools
import logging
import numpy as np
import imgaug as ia
from det3d.datasets import DEFAULT_TRANSFORMS, AUGMENTATION_TRANSFORMS
import math

img_dim = 224

# anchors must be in increasing order i.e. first 3 anchors are for small pedestrians
anchors = np.array([[12.8, 45.6], [18.6, 69.5], [23.5, 98.4], [32.8, 78.3], [31.8, 125.7], [52.4, 112.0], [41.7, 160.5],
                   [60.5, 164.1], [86.7, 160.0]], dtype=np.float32)  # for inria 224

contain_pcd = False
multiscale = True

# model settings
model = dict(
    type="DeepFusion",
    pretrained="C:/MSc_thesis/Pytorch_Codes/pretrained/mobilenetv3-large-1cd25616.pth",
    backbone_image=dict(
        type="MobileNetV3Large",
        num_input_features=3,
        width_mult=1.
    ),
    bbox_head=dict(
        type="YOLOv3Head",
        anchors=anchors,
        img_dim=img_dim,
        attention_channels=[104, 240, 960],
        first_map_conv_channels=[104, 128, 128, 64, 64, 32, 32, 16, 16, len(anchors) // 3 * 6],
        second_map_conv_channels=[240, 256, 256, 128, 128, 64, 64, 32, 32, len(anchors) // 3 * 6],
        third_map_conv_channels=[960, 512, 512, 256, 256, 128, 128, 64, 64, len(anchors) // 3 * 6],
        first_grid_channels=[104, 1],
        second_grid_channels=[240, 1],
        third_grid_channels=[960, 1],
    ),
)

assigner_cfg = dict(
    anchors=anchors,
    obj_thres=0.5,
    surrounding_size=4,
    top_k=1
)

test_cfg = dict(
    img_dim=img_dim,
    nms_thres=0.5,  # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.5,  # threshold to select between background or pedestrian
    iou_thres=0.5,  # threshold for matching predicted bounding boxes and groundtruths
    max_nms=30000,
    max_det=30,
)

# dataset settings
dataset_type = "InriaDataset"
nsweeps = 10

voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=[0.2, 0.2, 8],  # its length should be number of features of point clouds
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
)

train_anno = "C:/MSc_thesis/Dataset/inria_person/infos_train.pkl"
val_anno = "C:/MSc_thesis/Dataset/inria_person/infos_test.pkl"
version = "v1.0-mini"
test_anno = val_anno

data = dict(
    samples_per_gpu=20,  # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=train_anno,
        nsweeps=nsweeps,
        transform=AUGMENTATION_TRANSFORMS,
        multiscale=True,
        version=version,
    ),
    val=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        transform=DEFAULT_TRANSFORMS,
        version=version,
    ),
    test=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        transform=DEFAULT_TRANSFORMS,
    ),
)

# optimizer
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="AdamW", lr=0.001, weight_decay=3.0, paramwise_options=True, amsgrad=False)

# learning rate scheduler
lr_config = dict(
    type="OneCycleLR", max_lr=0.001, div_factor=25.0, pct_start=0.3,
)
# lr_config = dict(
#     type="linear_warmup",
#     warmup_learning_rate=1e-6,
#     warmup_steps=1000,
# )

checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)

# yapf:enable
# runtime settings
total_epochs = 400
log_level = "INFO"
pretrained = None
work_dir = './work_dirs/mobilenetv3_yolov3_inria/'
#resume_from = './work_dirs/mobilenetv3_yolov3_inria/epoch_120.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
