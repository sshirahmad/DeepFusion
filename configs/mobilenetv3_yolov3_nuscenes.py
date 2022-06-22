import itertools
import logging
import numpy as np
from det3d.datasets import DEFAULT_TRANSFORMS, AUGMENTATION_TRANSFORMS

tasks = [dict(num_class=1, class_names=["pedestrian"])]

# different image sizes has significant impact on results???
img_dim = 614  # 416 is another option

# anchors must be in increasing order i.e. first 3 anchors are for small pedestrians
# anchors = np.array([[5.2, 19.5], [7.8, 27.6], [11.3, 37.3], [15.7, 49.2], [21.3, 64.1], [27.6, 89.7], [38.6, 112.1],
#                     [56.1, 152.7], [97.7, 257.1]], dtype=np.float32)  # for img_dim=416
anchors = np.array([[2.81, 10.5], [4.2, 14.9], [6.1, 20.1], [8.4, 26.5], [11.5, 34.5], [14.8, 48.3], [20.8, 60.3],
                    [30.2, 82.2], [52.6, 138.4]], dtype=np.float32)  # for img_dim=224

contain_pcd = False
multiscale = False
pc_range = [-32.3, -0.2, 3.9, 20.7, 2.16, 92.8]
voxel_size = [0.16, 0.16, 8]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="DeepFusion",
    pretrained="C:/MSc_thesis/Pytorch_Codes/pretrained/mobilenetv3-large-1cd25616.pth",
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 3],
        num_input_features=5,
        with_distance=False,
        voxel_size=voxel_size,
        pc_range=pc_range,
    ),
    backbone_radar=dict(
        type="PointPillarsScatter",
        img_dim=img_dim,
        num_input_features=3
    ),
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
    max_per_img=500,
    img_dim=img_dim,
    nms_thres=0.5,   # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.1,  # threshold to select between background or pedestrian
    iou_thres=0.5,   # threshold for matching predicted bounding boxes and groundtruths
    max_nms=30000,
    max_det=10,
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10

voxel_generator = dict(
    range=pc_range,  # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=voxel_size,
    max_points_in_voxel=20,
    max_voxel_num=[12000, 12000],
)

train_anno = "C:/MSc_thesis/Dataset/NuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "C:/MSc_thesis/Dataset/NuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
version = "v1.0-trainval"
test_anno = val_anno

data = dict(
    samples_per_gpu=8,  # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=train_anno,
        nsweeps=nsweeps,
        voxel_cfg=voxel_generator,
        transform=DEFAULT_TRANSFORMS,
        shuffle_points=True,
        version=version,
    ),
    val=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        voxel_cfg=voxel_generator,
        transform=DEFAULT_TRANSFORMS,
        shuffle_points=True,
        version=version,
    ),
    test=dict(
        type=dataset_type,
        assigner_cfg=assigner_cfg,
        img_dim=img_dim,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        voxel_cfg=voxel_generator,
        transform=DEFAULT_TRANSFORMS,
        shuffle_points=True,
        version=version,
    ),
)

# optimizer
#optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="AdamW", lr=0.001, weight_decay=4.0, paramwise_options=True, amsgrad=False)
# optimizer = dict(type="SGD", lr=2e-4, momentum=0.9, weight_decay=5e-4, paramwise_options=True, nesterov=True)

# learning rate scheduler
lr_config = dict(
    type="OneCycleLR", max_lr=0.001, div_factor=25.0, pct_start=0.3,
)
# lr_config = dict(
#     type="ReduceLROnPlateau", mode='max', factor=0.5, patience=50, threshold=0.01, threshold_mode='rel'
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
total_epochs = 500
log_level = "INFO"
pretrained = None
work_dir = './work_dirs/mobilenetv3_yolov3_nuscenes/'
# resume_from = './work_dirs/mobilenetv3_yolov3_nuscenes/latest.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]

