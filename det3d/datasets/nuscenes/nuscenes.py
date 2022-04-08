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
import torch
import cv2
from det3d.core.utils import yolo_utils
from matplotlib.ticker import NullLocator
import os

from functools import reduce
from pathlib import Path
from copy import deepcopy

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    get_sample_data,
    _deepfusion_det_to_nusc_box,
    _cam_nusc_box_to_global,
    eval_main
)
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    NumPointFeatures = 5

    def __init__(
            self,
            info_path,
            root_path,
            nsweeps=0,  # here set to zero to catch unset nsweep
            cfg=None,
            pipeline=None,
            class_names=None,
            test_mode=False,
            version="v1.0-trainval",
            **kwargs,
    ):
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print("number of sweeps: ", self.nsweeps)

        self._info_path = info_path
        self._class_names = class_names

        if not hasattr(self, "_nusc_infos"):
            self.load_infos()

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def load_infos(self):
        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

            self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos()

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]

            # get pedestrian class groundtruths
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            mask &= np.array([n == "pedestrian" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]

            # discard boxes outside of detection range
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)

            # conver 3D boxes to 2D boxes and resize them
            intrinsic = info["cam_intrinsic"]

            gt_annos.append(
                {
                    "boxes": gt_boxes[mask],
                    "cam_intrinsic": intrinsic,
                    "image": Path(info["cam_front_path"]),
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx, input_size):

        info = self._nusc_infos[idx]

        res = {
            "radar": {
                "type": "radar",
                "points": None,
                "nsweeps": self.nsweeps,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            # "calib": None,
            "camera": {
                "type": "camera",
                "image": None,
                "image_resize": input_size,
                "annotations": None,
            },
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx, input_size = idx
        else:
            # set the default image size here
            input_size = None

        return self.get_sensor_data(idx, input_size)

    def evaluation(self, detections, iou_thres=0.5, img_dim=416, output_dir=None, val_mode=False):
        # calculate AP for validation set
        gt_annos = self.ground_truth_annotations
        assert gt_annos is not None

        # Calculate AP
        sample_metrics = []
        cmatched_boxes = [None for _ in range(len(gt_annos))]
        cmatched_scores = [None for _ in range(len(gt_annos))]
        labels = []
        miss = 0
        for i, gt in enumerate(gt_annos):
            if detections[gt["token"]]["boxes"] is not None:
                img = np.array(Image.open(gt["image"]).convert('RGB'), dtype=np.uint8)
                # convert 3D boxes to 2D boxes
                intrinsic = gt["cam_intrinsic"]
                minmax_boxes = box_np_ops.convert_box3d_box2d(gt["boxes"], intrinsic).reshape(-1, 4)
                _, norm_corner_bbox = prep.normalize_data(img, minmax_boxes, size=img_dim)
                target_boxes = torch.tensor(norm_corner_bbox * img_dim)

                # match predicted boxes with groundtruths
                metrics, boxes, scores = yolo_utils.match_boxes(detections[gt["token"]], target_boxes, iou_threshold=iou_thres)
                sample_metrics.append(metrics[0])
                labels += target_boxes.tolist()
                if boxes:
                    cmatched_boxes[i] = boxes
                    cmatched_scores[i] = scores
            else:
                miss += len(gt["boxes"])

        # Concatenate sample statistics
        matched_boxes = [None for _ in range(len(gt_annos))]
        for i, x in enumerate(cmatched_boxes):
            if x is not None:
                matched_boxes[i] = torch.cat(x, 0)

        matched_scores = [None for _ in range(len(gt_annos))]
        for i, x in enumerate(cmatched_scores):
            if x is not None:
                matched_scores[i] = torch.cat(x, 0)

        if len(sample_metrics) != 0:
            true_positives, pred_scores = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision_curve, recall_curve, precision, recall, AP, f1 = yolo_utils.compute_ap(
                true_positives, pred_scores, labels, miss
            )
        else:
            precision_curve = torch.tensor([])
            recall_curve = torch.tensor([])
            precision = 0
            recall = 0
            AP = 0
            f1 = 0

        lamr, mr, fppi = yolo_utils.log_average_miss_rate(precision_curve, recall_curve)

        if val_mode:
            result_dict = {
                "AP": AP * 100,
                "F1 Score": f1 * 100,
                "Recall": recall * 100,
                "Precision": precision * 100,
                "MR": lamr * 100
            }
            return result_dict
        else:
            print(f"\nAverage Precision:{AP}\n Recall:{recall}\n Precision:{precision}\n F1 score:{f1}\n Log average "
                  f"miss rate:{lamr}")

            result_path = os.path.join(output_dir, "Results")
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_name = os.path.join(output_dir, "Results", "recall_precision.png")
            # Precision-Recall curve
            mrec = np.concatenate(([0.0], recall_curve, [1.0]))
            mpre = np.concatenate(([0.0], precision_curve, [0.0]))

            plt.figure()
            plt.plot(mrec, mpre)
            plt.ylabel("Precision")
            plt.xlabel("Recall")
            plt.grid()
            plt.savefig(file_name)
            plt.close()

            file_name = os.path.join(output_dir, "Results", "miss_fppi.png")
            fppi_tmp = np.insert(fppi, 0, -1.0)
            mr_tmp = np.insert(mr, 0, 1.0)

            plt.figure()
            plt.plot(fppi_tmp, mr_tmp)
            plt.ylabel("Miss Rate")
            plt.xlabel("FPPI")
            plt.grid()
            plt.savefig(file_name)
            plt.close()

            img_detections = [None for _ in range(len(gt_annos))]  # Stores detections for each image index
            for i, (boxes, scores) in enumerate(zip(matched_boxes, matched_scores)):
                # Save image and detections
                if boxes is not None:
                    img_detections[i] = torch.cat((boxes, scores.unsqueeze(-1)), dim=-1)

            image_path = os.path.join(output_dir, "DeepFusion")
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            print("\nSaving images:")
            # Iterate through images and save plot of detections
            for img_i, (im_det, gt) in enumerate(zip(img_detections, gt_annos)):
                path = gt["image"]

                print("(%d) Image: '%s'" % (img_i, path))

                # Create plot
                img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                # Bounding-box colors
                cmap = plt.get_cmap("tab20")
                colors = [cmap(i) for i in np.linspace(0, 1, 20)]
                color_pred = colors[6]
                color_gt = colors[4]
                # Draw bounding boxes and labels of detections
                intrinsic = gt["cam_intrinsic"]
                minmax_boxes = box_np_ops.convert_box3d_box2d(gt["boxes"], intrinsic).reshape(-1, 4)
                for x1_gt, y1_gt, x2_gt, y2_gt in minmax_boxes:
                    w_gt = x2_gt - x1_gt
                    h_gt = y2_gt - y1_gt

                    bbox_gt = patches.Rectangle((x1_gt, y1_gt), w_gt, h_gt, linewidth=2, edgecolor=color_gt,
                                                facecolor="none")
                    ax.add_patch(bbox_gt)
                if im_det is not None:
                    # Rescale boxes to original image
                    im_det[:, :4] = yolo_utils.rescale_boxes(im_det[:, :4], img_dim, img.shape[:2])
                    for x1_pred, y1_pred, x2_pred, y2_pred, conf_score in im_det:
                        w_pred = x2_pred - x1_pred
                        h_pred = y2_pred - y1_pred

                        # Create a Rectangle patch
                        bbox_pred = patches.Rectangle((x1_pred, y1_pred), w_pred, h_pred, linewidth=2,
                                                      edgecolor=color_pred, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox_pred)
                        # Add label
                        plt.text(
                            x1_pred,
                            y1_pred,
                            s=conf_score.numpy().round(2),
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color_pred, "pad": 0},
                        )

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                filename = os.path.basename(path).split(".")[0]
                output_path = os.path.join(image_path, f"{filename}.png")
                plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
                plt.close()

            images = [img for img in os.listdir(image_path) if img.endswith(".png")]
            frame = cv2.imread(os.path.join(image_path, images[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(os.path.join(image_path, "video.avi"), fourcc, 2, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(image_path, image)))

            cv2.destroyAllWindows()
            video.release()

        return None
