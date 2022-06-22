import itertools
import logging
import numpy as np
from det3d.datasets import DEFAULT_TRANSFORMS, AUGMENTATION_TRANSFORMS

# different image sizes has significant impact on results???
img_dim = 224  # 416 is another option

# anchors must be in increasing order i.e. first 3 anchors are for small pedestrians
# anchors = np.array([[14.2, 51.1], [20.7, 62.6], [25.5, 87.9], [43.6, 60.4], [38.8, 127.5],
#                     [97.5, 71.4], [63.7, 205.1], [213.2, 83.5], [103.5, 309.5]],
#                    dtype=np.float32)  # for img_dim=416
anchors = np.array([[7.7, 27.5], [11.1, 33.7], [13.7, 47.3], [23.5, 32.5], [20.9, 68.6],
                    [52.5, 38.5], [34.3, 110.4], [114.8, 44.9], [55.7, 166.7]],
                   dtype=np.float32)  # for img_dim=224

contain_pcd = False
multiscale = False

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
        attention_channels=[72, 144, 960],
        first_map_conv_channels=[72, 128, 128, 64, 64, 32, 32, 16, 16, len(anchors) // 3 * 5],
        second_map_conv_channels=[144, 128, 128, 64, 64, 32, 32, 16, 16, len(anchors) // 3 * 5],
        third_map_conv_channels=[960, 128, 128, 64, 64, 32, 32, 16, 16, len(anchors) // 3 * 5],
        first_grid_channels=[72, 1],
        second_grid_channels=[144, 1],
        third_grid_channels=[960, 1],
    ),
)


test_cfg = dict(
    max_per_img=500,
    img_dim=img_dim,
    nms_thres=0.5,   # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.1,  # threshold to select between background or pedestrian
    iou_thres=0.5,   # threshold for matching predicted bounding boxes and groundtruths
    max_nms=30000,
    max_det=10,
)

# dataset settings
dataset_type = "CaltechDataset"
nsweeps = 10

voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=[0.2, 0.2, 8],  # its length should be number of features of point clouds
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
)

train_anno = "/datadrive/Dataset/Caltech/infos_train.pkl"
val_anno = "/datadrive/Dataset/Caltech/infos_test.pkl"
version = "v1.0-mini"
test_anno = val_anno

data = dict(
    samples_per_gpu=20,  # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_dim=img_dim,
        info_path=train_anno,
        nsweeps=nsweeps,
        version=version,
        transform=AUGMENTATION_TRANSFORMS,
    ),
    val=dict(
        type=dataset_type,
        img_dim=img_dim,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        version=version,
        transform=DEFAULT_TRANSFORMS,
    ),
    test=dict(
        type=dataset_type,
        img_dim=img_dim,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        transform=DEFAULT_TRANSFORMS,
    ),
)

# optimizer
# optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01, paramwise_options=False, amsgrad=False)
# optimizer = dict(type="SGD", lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4, paramwise_options=True)

# learning rate scheduler
lr_config = dict(
    type="OneCycleLR", max_lr=0.001, div_factor=25.0, pct_start=0.3,
)
# lr_config = dict(
#     type="cosine_warmup", warmup_learning_rate=1e-6, warmup_steps=1,
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
total_epochs = 300
log_level = "INFO"
pretrained = None
work_dir = './work_dirs/mobilenetv3_yolov3_caltech/'
# resume_from = './work_dirs/mobilenetv3_yolov3_caltech/latest.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
