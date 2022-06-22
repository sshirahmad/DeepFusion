import numpy as np
from pathlib import Path
from det3d.core import box_np_ops
from ..registry import PIPELINES
from nuscenes.utils.geometry_utils import view_points
import cv2


@PIPELINES.register_module
class LoadSensorsData(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":
            points = info["sweeps"].T
            times = info["sweep_times"].astype(points.dtype)

            points = points.T[:, [0, 1, 2, 8, 9]]
            res["radar"]["points"] = points
            res["radar"]["times"] = times
            res["radar"]["combined"] = np.hstack([points, times])

            cam_path = Path(info["cam_front_path"])
            res["camera"]["image"] = cam_path

        elif self.type in ["CaltechDataset", "InriaDataset"]:
            cam_path = info["image_path"].rstrip()
            res["camera"]["image"] = cam_path

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadAnnotations(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0

            # convert 3D boxes to 2D boxes
            intrinsic = info["cam_intrinsic"]
            gt_boxes = box_np_ops.convert_box3d_box2d(gt_boxes, intrinsic).reshape(-1, 4)

            res["camera"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }

        elif res["type"] in ["CaltechDataset", "InriaDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0

            res["camera"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
            }
        else:
            pass

        return res, info
