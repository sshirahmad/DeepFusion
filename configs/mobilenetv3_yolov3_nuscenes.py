import itertools
import logging
import numpy as np
import imgaug.augmenters as iaa
from det3d.utils.config_tool import get_downsample_factor

tasks = [dict(num_class=1, class_names=["pedestrian"])]

# different image sizes has significant impact on results???
img_dim = 224  # 416 is another option

# anchors must be in increasing order i.e. first 3 anchors are for small pedestrians
# anchors = np.array([[5.2, 19.5], [7.8, 27.6], [11.3, 37.3], [15.7, 49.2], [21.3, 64.1], [27.6, 89.7], [38.6, 112.1],
#                     [56.1, 152.7], [97.7, 257.1]], dtype=np.float32)  # for img_dim=416
anchors = np.array([[2.81, 10.5], [4.2, 14.9], [6.1, 20.1], [8.4, 26.5], [11.5, 34.5], [14.8, 48.3], [20.8, 60.3],
                    [30.2, 82.2], [52.6, 138.4]], dtype=np.float32)  # for img_dim=224

contain_pcd = True
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
    pretrained=None,
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
        type="VGG16",
        num_input_features=3+3,
        width_mult=1.
    ),
    bbox_head=dict(
        type="YOLOv3Head",
        tasks=tasks,
        anchors=anchors,
        img_dim=img_dim,
        attention_channels=[168, 368, 960],  # Shouldn't be changed
        first_map_conv_channels=[168, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16, len(anchors) // 3 * 6,
                                 len(anchors) // 3 * 6],  # shouldn't change the first one (40)
        second_map_conv_channels=[368, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, len(anchors) // 3 * 6,
                                  len(anchors) // 3 * 6],  # shouldn't change the first one (112)
        third_map_conv_channels=[960, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, len(anchors) // 3 * 6,
                                 len(anchors) // 3 * 6],  # shouldn't change the first one
        first_grid_channels=[168, 1],
        second_grid_channels=[368, 1],
        third_grid_channels=[960, 1],
        noobj_scale=1,
        obj_scale=1,
        zero_scale=1,
        value_scale=1,
        alpha=0.8,
        gamma=2,
        exp_alpha=2,
        num_classes=1,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    ignore_thres=0.5,
    img_dim=img_dim,
    surrounding_size=4,
    top_k=2,
    anchors=anchors,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    max_per_img=500,
    img_dim=img_dim,
    nms_thres=0.6,   # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.5,  # threshold to select between background or pedestrian
    iou_thres=0.5,   # threshold for matching predicted bounding boxes and groundtruths
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "E:/MSc/Thesis/Dataset/Nuscenes/nuScenes/"

db_sampler = None

train_preprocessor = dict(
    mode="train",
    img_size=img_dim,
    shuffle_points=True,
    contain_pcd=contain_pcd,
    global_rot_noise=[-0.4363, 0.4363],
    global_scale_noise=[0.8, 1.2],
    global_translate_noise=44,
    image_aug_seq=[
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ],
    db_sampler=None,
    no_augmentation=False,
    class_names=class_names,
)

test_preprocessor = dict(
    mode="val",
    contain_pcd=contain_pcd,
    img_size=img_dim,
    shuffle_points=False,
)

voxel_generator = dict(
    range=pc_range,  # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=voxel_size,
    max_points_in_voxel=20,
    max_voxel_num=[12000, 12000],
)

train_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabelYOLOv3", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

test_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=test_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabelYOLOv3", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "E:/MSc/Thesis/Dataset/Nuscenes/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "E:/MSc/Thesis/Dataset/Nuscenes/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
version = "v1.0-trainval"
test_anno = val_anno

data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        version=version,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        ann_file=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        version=version,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
#optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01, paramwise_options=True, amsgrad=False)
# optimizer = dict(type="SGD", lr=2e-4, momentum=0.9, weight_decay=5e-4, paramwise_options=True, nesterov=True)

# learning rate scheduler
lr_config = dict(
    type="OneCycleLR", max_lr=0.001, div_factor=25.0, pct_start=0.3,
)
# lr_config = dict(
#     type="ReduceLROnPlateau", mode='max', factor=0.5, patience=50, threshold=0.01, threshold_mode='rel'
# )

checkpoint_config = dict(interval=1)
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
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/mobilenetv3_yolov3_nuscenes/'
# resume_from = './work_dirs/mobilenetv3_yolov3_nuscenes/latest.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]

