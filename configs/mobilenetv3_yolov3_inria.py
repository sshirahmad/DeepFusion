import itertools
import logging
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import math

ia.seed(1)

tasks = [dict(num_class=1, class_names=["pedestrian"])]

# different image sizes has significant impact on results???
img_dim = 224

# ratio = [1, 1 / 2, 1 / 3]
# scale = [0.2, 0.55, 0.9]
# anchors = []
# for s in scale:
#     for r in ratio:
#         anchors.append([s * math.sqrt(r), s / math.sqrt(r)])
#
# anchors = np.array(anchors, dtype=np.float32)
# anchors must be in increasing order i.e. first 3 anchors are for small pedestrians
# anchors = np.array([[27, 95], [38, 144], [60, 151], [48, 203], [65, 259], [89, 207], [91, 325],
#                     [126, 256], [173, 315]],
#                    dtype=np.float32)  # for inria 448
anchors = np.array([[12.8, 45.6], [18.6, 69.5], [23.5, 98.4], [32.8, 78.3], [31.8, 125.7], [52.4, 112.0], [41.7, 160.5],
                    [60.5, 164.1], [86.7, 160.0]], dtype=np.float32)  # for inria 224

contain_pcd = False
multiscale = True
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="DeepFusion",
    pretrained=None,
    backbone_image=dict(
        type="MobileNetV3Large",
        num_input_features=3,
        width_mult=1.
    ),
    bbox_head=dict(
        type="YOLOv3Head",
        tasks=tasks,
        anchors=anchors,
        img_dim=img_dim,
        attention_channels=[104, 176, 960],  # Shouldn't be changed
        first_map_conv_channels=[104, 256, 256, 128, 128, 64, 64, 32, 32, len(anchors) // 3 * 6],
        second_map_conv_channels=[176, 256, 256, 128, 128, 64, 64, 32, 32, len(anchors) // 3 * 6],
        third_map_conv_channels=[960, 256, 256, 128, 128, 64, 64, 32, 32, len(anchors) // 3 * 6],
        first_grid_channels=[104, 1],
        second_grid_channels=[176, 1],
        third_grid_channels=[960, 1],
        alpha=0.25,
        gamma=2,
        exp_alpha=2,
        num_classes=1,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    obj_thres=0.5,
    top_k=1,
    img_dim=img_dim,
    surrounding_size=4,
    anchors=anchors,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    img_dim=img_dim,
    nms_thres=0.5,  # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.5,  # threshold to select between background or pedestrian
    iou_thres=0.5,  # threshold for matching predicted bounding boxes and groundtruths
)

# dataset settings
dataset_type = "InriaDataset"
nsweeps = 10
data_root = "C:/MSc_thesis/Dataset/inria_person"

db_sampler = None

train_preprocessor = dict(
    mode="train",
    img_size=img_dim,
    shuffle_points=True,
    contain_pcd=contain_pcd,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    image_aug_seq=[
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.2)),  # random crops
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
        )
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
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=[0.2, 0.2, 8],  # its length should be number of features of point clouds
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
)

train_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabelYOLOv3", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

test_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=test_preprocessor),
    dict(type="AssignLabelYOLOv3", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "C:/MSc_thesis/Dataset/inria_person/infos_train.pkl"
val_anno = "C:/MSc_thesis/Dataset/inria_person/infos_test.pkl"
version = "v1.0-mini"
test_anno = val_anno

data = dict(
    samples_per_gpu=20,  # batch size
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=val_anno,
        nsweeps=train_anno,
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
        info_path=val_anno,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01, paramwise_options=True, amsgrad=False)

# learning rate scheduler
lr_config = dict(
    type="OneCycleLR", max_lr=0.001, div_factor=25.0, pct_start=0.3,
)
# lr_config = dict(
#     type="cosine_warmup",
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
total_epochs = 4000
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
pretrained = None
work_dir = './work_dirs/mobilenetv3_yolov3_inria/'
# resume_from = './work_dirs/mobilenetv3_yolov3_inria/epoch_7600.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
