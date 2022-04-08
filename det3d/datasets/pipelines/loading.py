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

            nsweeps = res["radar"]["nsweeps"]
            points = info["sweeps"].T
            times = info["sweep_times"].astype(points.dtype)

            image = cv2.imread(info["cam_front_path"])
            height, width, channel = image.shape

            # Grab the depths (camera frame z axis points away from the camera).
            depths = points[2, :]

            # convert 3D point clouds to 2D image points
            point_indices = points[[0, 1, 2], :]
            point_indices = view_points(point_indices, info["cam_intrinsic"], normalize=True)[:2]

            # mask points that are out of image
            x = point_indices[0, :]
            y = point_indices[1, :]
            mask = depths > 0
            mask &= x < width
            mask &= 0 < x
            mask &= y < height
            mask &= 0 < y
            point_indices = point_indices[:, mask]
            points = points[:, mask]
            times = times[mask]

            # Form radar images
            # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
            # vx, vy are the velocities in m/s.
            # vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
            # We recommend using the compensated velocities.
            radar_image = np.zeros(shape=(height, width, channel), dtype=np.uint8)
            # TODO could be changed to use other features of point clouds
            radar_image[point_indices[1, :].astype(np.int_), point_indices[0, :].astype(np.int_), :] = points[[2, 8, 9], :].T

            "visualize sparse radar images"
            # resized_radar_image = cv2.resize(radar_image, (416, 416))
            # for i, p in enumerate(point_indices.T):
            #     img = cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
            # resized_image = cv2.resize(img, (416, 416))
            # cv2.imwrite('radar_image.png', resized_radar_image * 255)
            # cv2.imwrite('image.png', resized_image)
            # cv2.waitKey(0)

            points = points.T[:, [0, 1, 2, 8, 9]]
            res["radar"]["image"] = radar_image
            res["radar"]["points"] = points
            res["radar"]["times"] = times
            res["radar"]["combined"] = np.hstack([points, times])

            cam_path = Path(info["cam_front_path"])
            res["camera"]["image"] = cam_path

        elif self.type == "CaltechDataset" or self.type == "InriaDataset":

            cam_path = Path(info["image_path"])
            res["camera"]["image"] = cam_path

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["radar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }

            res["camera"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }

        elif res["type"] in ["CaltechDataset", "InriaDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            
            xmin = gt_boxes[:, 0] - gt_boxes[:, 2] / 2.0
            ymin = gt_boxes[:, 1] - gt_boxes[:, 3] / 2.0
            xmax = gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0
            ymax = gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0

            minmax_boxes = np.stack((xmin, ymin, xmax, ymax), axis=-1)
            res["camera"]["annotations"] = {
                "boxes": minmax_boxes,
                "names": info["gt_names"],
            }
        else:
            pass

        return res, info
