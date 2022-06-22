import itertools
import logging
import numpy as np
import imgaug.augmenters as iaa
from det3d.utils.config_tool import get_downsample_factor

tasks = [dict(num_class=1, class_names=["pedestrian"])]

# different image sizes has significant impact on results???
img_dim = 448  # 416 is another option

anchors = np.array([[42, 149], [99, 265]], dtype=np.float32)  # for inria 448
# anchors = np.array([[47, 134], [22, 72]], dtype=np.float32)  # for inria 224

grid_maps = [56, 28, 14]  # for img_dim=448
# grid_maps = [28, 14, 7]  # for img_dim=224

contain_pcd = False
multiscale = False

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
        type="MobileNetV2",
        num_input_features=3,
        width_mult=1.
    ),
    bbox_head=dict(
        type="YOLOHead",
        tasks=tasks,
        anchors=anchors,
        grid_maps=grid_maps,
        img_dim=img_dim,
        attention_channels=[256, 192, 640],  # Shouldn't be changed
        feature_map_channels=[32, 96, 1280],  # Shouldn't be changed
        last_conv_channels=[1088, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32,
                            len(anchors) * 6, len(anchors) * 6],  # shouldn't change the first one
        first_grid_channels=[32, 128, 64, 1],
        second_grid_channels=[96, 256, 128, 1],
        third_grid_channels=[1280, 512, 256, 1],
        noobj_scale=1,
        obj_scale=1,
        zero_scale=1,
        value_scale=1,
        num_classes=1,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    ignore_thres=0.7,
    grid_maps=grid_maps,  # Shouldn't be changed
    img_dim=img_dim,
    surrounding_size=4,
    top_k=2,
    anchors=anchors,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    keep_top_k=20,
    img_dim=img_dim,
    nms_thres=0.6,   # threshold for merging overlapping predicted bounding boxes
    conf_thres=0.5,  # threshold to select between background or pedestrian
    iou_thres=0.5,   # threshold for matching predicted bounding boxes and groundtruths
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
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
        )],
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
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # (x_min, y_min, z_min, x_max, y_max, z_max) and minmax of other point cloud features
    voxel_size=[0.2, 0.2, 8],  # its length should be number of features of point clouds
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
)

train_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabelYOLO", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

test_pipeline = [
    dict(type="LoadSensorsData", dataset=dataset_type),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=test_preprocessor),
    dict(type="AssignLabelYOLO", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "C:/MSc_thesis/Dataset/inria_person/infos_train.pkl"
val_anno = "C:/MSc_thesis/Dataset/inria_person/infos_test.pkl"
version = "v1.0-mini"
test_anno = val_anno

data = dict(
    samples_per_gpu=10,  # batch size
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
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))  # prevents gradient explosion!
optimizer_config = None
optimizer = dict(type="Adam", lr=0.001, weight_decay=1e-4, paramwise_options=True, amsgrad=False)
# optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=5e-4)

# learning rate scheduler
# lr_config = dict(
#     type="OneCycleLR", max_lr=0.001, div_factor=10.0, pct_start=0.4,
# )
lr_config = dict(
    type="linear_warmup", warmup_learning_rate=1e-6, warmup_steps=1000
)

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
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
pretrained = None
work_dir = './work_dirs/mobilenetv2_yolo_inria/'
# resume_from = './work_dirs/mobilenetv2_yolo_inria/epoch_30.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
