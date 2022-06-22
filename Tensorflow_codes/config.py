import numpy as np
import math

dataset_name = 'inria_person'  # can be 'hazy_person or inria_person'
category_index = {0: {"name": "Background"},
                  1: {"name": "Person"}}

img_size = 224
grid_size = list([img_size // 8, img_size // 16, img_size // 32])

# ratio = [1, 1 / 2, 1 / 3]
# scale = [0.2, 0.55, 0.9]
# anchors = []
# for s in scale:
#     for r in ratio:
#         anchors.append([s / math.sqrt(r), s * math.sqrt(r)])
#
# anchors = np.array(anchors, dtype=np.float32)
anchors = np.array([[45.6, 12.8], [69.5, 18.6], [98.4, 23.5], [78.3, 32.8], [125.7, 31.8], [112.0, 52.4], [160.5, 41.7],
                    [164.1, 60.5], [160.0, 86.7]], dtype=np.float32)  # for inria 224
# anchors=np.array([[95, 27], [144, 38], [151, 60], [203, 48], [259, 65], [207, 89], [325, 91],
#                   [256, 126], [315, 173]], dtype=np.float32)  # for inria 448

train = dict(
    neg_ratio=3.0,
    model_name="DeepFusion",
    backbone_name="mobilenet_v3",
)

logger = dict(
    log_step=20,
    summary_path='./summary/',
    summary_step=20,
)

checkpoint = dict(
    save_step=500,
    checkpoint_path='./checkpoint/',

)

test_config = dict(
    conf_threshold=0.5,
    nms_threshold=0.5,
    iou_threshold=0.5,
    keep_top_k=200,
)

lr_scheduler = dict(
    type='cosine_warmup',
    warmup_step=1000,
    warmup_init=1e-6,
    max_lr=0.001,
    max_wd=5e-4,
)

dataset = dict(
    type='inria',
    anchors=anchors,
    batch_size=20,
    obj_threshold=0.5,
    top_k=1,
    surounding_size=4,
    img_size=img_size,
    grid_size=grid_size,
    train_dir="C:/MSc_thesis/Dataset/inria_person/infos_train.pkl",
    val_dir="C:/MSc_thesis/Dataset/inria_person/infos_test.pkl",

)

output_dir = './work_dir/'
training_step = 35000
validation = True
