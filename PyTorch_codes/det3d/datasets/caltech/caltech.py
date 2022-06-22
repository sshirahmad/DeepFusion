import sys
import pickle
import json
import random
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from PIL import Image
from det3d.core import xywh2xyxy
from det3d.datasets.transforms import resize
from torch.utils.data import Dataset
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class CaltechDataset(Dataset):
    def __init__(
            self,
            img_dim,
            info_path,
            transform=None,
            test_mode=False,
            **kwargs,
    ):
        self._info_path = info_path
        self.test_mode = test_mode
        self.transform = transform
        self.img_dim = img_dim

        if not hasattr(self, "_caltech_infos"):
            self.load_infos()

    def load_infos(self):
        with open(self._info_path, "rb") as f:
            _caltech_infos_all = pickle.load(f)

        self._caltech_infos = _caltech_infos_all

    def __len__(self):

        if not hasattr(self, "_caltech_infos"):
            self.load_infos()

        return len(self._caltech_infos)

    def get_sensor_data(self, idx, input_size):
        # ---------
        #  Image
        # ---------
        try:
            img_path = self._caltech_infos[idx % len(self._caltech_infos)]['image_path'].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        boxes = self._caltech_infos[idx % len(self._caltech_infos)]['gt_boxes']

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        # Create grid classifiers targets
        nG = list([self.img_dim // 8, self.img_dim // 16, self.img_dim // 32])
        corner_boxes = xywh2xyxy(bb_targets[:, 1:]).numpy()
        grid_maps = []
        for k, size in enumerate(nG):
            grid_map = np.zeros(shape=(size, size), dtype=np.float32)
            x = np.arange(size, dtype=np.float32)
            y = x.T
            if corner_boxes.any():
                for i in range(size):
                    for j in range(size):
                        box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                        iou = box_np_ops.overlap_jit(corner_boxes * size, box)
                        grid_map[j, i] = iou.max(0)

            grid_maps.append(grid_map)

        img = resize(img, input_size)
        data_bundle = dict(
            image=img,
            image_path=img_path,
        )
        data_bundle.update({'gt_boxes': bb_targets,
                            'classifier_map1': grid_maps[0],
                            'classifier_map2': grid_maps[1],
                            'classifier_map3': grid_maps[2],
                            })

        return data_bundle

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx, input_size = idx
        else:
            # set the default image size here
            input_size = self.img_dim

        return self.get_sensor_data(idx, input_size)


