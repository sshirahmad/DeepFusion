import sys
import pickle
import json
import random
from collections import defaultdict
import numpy as np
import torch
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.core.utils.yolo_utils import xywh2xyxy_np
from PIL import Image
from det3d.core.utils.yolo_utils import build_targets
from det3d.datasets.transforms import resize
from torch.utils.data import Dataset
from det3d.datasets.registry import DATASETS
from .point_cloud_ops import Voxelization
import cv2


@DATASETS.register_module
class NuScenesDataset(Dataset):
    def __init__(
            self,
            img_dim,
            info_path,
            assigner_cfg,
            voxel_cfg,
            transform=None,
            multiscale=False,
            shuffle_points=False,
            test_mode=False,
            **kwargs,
    ):
        self._info_path = info_path
        self.test_mode = test_mode
        self.transform = transform
        self.img_size = img_dim
        self.assigner = assigner_cfg
        self.voxelize = Voxelization(cfg=voxel_cfg)
        self.shuffle_points = shuffle_points
        self.min_points_in_gt = kwargs.get("min_points_in_gt", -1)
        self.multiscale = multiscale and not test_mode
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        if not hasattr(self, "_nuscenes_infos"):
            self.load_infos()

    def load_infos(self):
        with open(self._info_path, "rb") as f:
            _nuscenes_infos_all = pickle.load(f)

        self._nuscenes_infos = _nuscenes_infos_all

    def __len__(self):

        if not hasattr(self, "_nuscenes_infos"):
            self.load_infos()

        return len(self._nuscenes_infos)

    def get_sensor_data(self, idx):
        # ---------
        #  Load Images and Rada
        # ---------
        try:
            points = self._nuscenes_infos[idx % len(self._nuscenes_infos)]["sweeps"].T
            points = points.T[:, [0, 1, 2, 8, 9]]  # x, y, z, vx, and vy
            if self.shuffle_points:
                np.random.shuffle(points)

            img_path = self._nuscenes_infos[idx % len(self._nuscenes_infos)]["image_path"].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read data '{img_path}'.")
            return

        # ---------
        #  Loading Groundtruths
        # -----------
        gt_boxes = self._nuscenes_infos[idx % len(self._nuscenes_infos)]["gt_boxes"].astype(np.float32)
        gt_boxes[np.isnan(gt_boxes)] = 0

        # Discard boxes that include few radar point clouds
        if self.min_points_in_gt > 0:
            point_counts = box_np_ops.points_count_rbbox(points, gt_boxes)
            mask = point_counts >= self.min_points_in_gt
            gt_boxes = gt_boxes[mask]

        # convert 3D boxes to 2D boxes
        intrinsic = self._nuscenes_infos[idx % len(self._nuscenes_infos)]["cam_intrinsic"]
        gt_boxes = box_np_ops.convert_box3d_box2d(gt_boxes, intrinsic).reshape(-1, 4)
        boxes = np.zeros((len(gt_boxes), 4))
        boxes[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        boxes[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
        boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

        # ---------
        #  Voxelization
        # -----------
        voxel_dict = self.voxelize(points)

        # -----------
        #  Augmentation and Normalization
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        # Create grid classifiers targets
        target_dict = build_targets(bb_targets.numpy(), self.img_size,
                      self.assigner.anchors, self.assigner.surrounding_size,
                      self.assigner.top_k, self.assigner.obj_thres)

        img = resize(img, self.img_size)
        data_bundle = dict(
            image=img,
            points=points,
            image_path=img_path,
            gt_boxes=bb_targets,

        )

        "visualize preprocessed ground truths"
        # image = np.moveaxis(img.numpy(), 0, -1).copy()
        # for box in bb_targets.numpy():
        #     box = xywh2xyxy_np((box[1:])) * self.img_size
        #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # data_bundle.update(voxel_dict)
        # data_bundle.update(target_dict)

        return data_bundle

    def __getitem__(self, idx):

        return self.get_sensor_data(idx)

    def collate_fn(self, batch_list):
        self.batch_count += 1

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        example_merged = defaultdict(list)
        for example in batch_list:
            if type(example) is list:
                for subexample in example:
                    for k, v in subexample.items():
                        example_merged[k].append(v)
            else:
                for k, v in example.items():
                    example_merged[k].append(v)
        ret = {}
        for key, elems in example_merged.items():
            if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels"]:
                ret[key] = torch.tensor(np.concatenate(elems, axis=0))

            elif key == "metadata":
                ret[key] = elems

            elif key in ["coordinates", "points"]:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                    )
                    coors.append(coor_pad)
                ret[key] = torch.tensor(np.concatenate(coors, axis=0))

            elif key == "gt_boxes":
                # Add sample index to targets
                for i, boxes in enumerate(elems):
                    boxes[:, 0] = i
                ret[key] = torch.cat(elems, 0)

            elif key in ["yolo_map1", "yolo_map2", "yolo_map3",
                         "classifier_map1", "classifier_map2", "classifier_map3",
                         "obj_mask1", "obj_mask2", "obj_mask3", "noobj_mask1", "noobj_mask2",
                         "noobj_mask3"]:
                ret[key] = torch.tensor(np.stack(elems, axis=0))
            elif key in ["image"]:
                ret[key] = torch.stack(elems)
            else:
                ret[key] = np.stack(elems, axis=0)
        return ret


