import abc
import sys
import time
from collections import OrderedDict
from functools import reduce
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import numba
import numpy as np

from det3d.core.bbox import box_np_ops


def global_scaling(points, scale=0.05):
    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale

    return points, noise_scale


def global_translate(points, noise_translate_std):
    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [-noise_translate_std, noise_translate_std]
        )
    if all([e == 0 for e in noise_translate_std]):
        return points, np.array([0, 0, 0]).T

    noise_translate = np.array(
        [
            np.random.uniform(noise_translate_std[0], noise_translate_std[1]),
            np.random.uniform(noise_translate_std[0], noise_translate_std[1]),
            np.random.uniform(noise_translate_std[0], noise_translate_std[1]),
        ]
    ).T

    points[:, :3] += noise_translate

    return points, noise_translate


def global_rotation(points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])  # in radians
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )  # rotation along the axis pointing inside the image (z axis)

    return points, noise_rotation * 180 / np.pi


def random_flip(points, probability=0.5):
    enable = np.random.choice([False, True], replace=False, p=[1 - probability, probability])
    if enable:
        points[:, 1] = -points[:, 1]  # flip along the horizontal axis (y axis)

    return points, enable


def random_flip_both(gt_boxes, points, probability=0.5, flip_coor=None):
    # x flip
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    # y flip
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        if flip_coor is None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            points[:, 0] = -points[:, 0]
        else:
            gt_boxes[:, 0] = flip_coor * 2 - gt_boxes[:, 0]
            points[:, 0] = flip_coor * 2 - points[:, 0]

        gt_boxes[:, -1] = -gt_boxes[:, -1] + 2 * np.pi  # TODO: CHECK THIS

        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 6] = -gt_boxes[:, 6]

    return gt_boxes, points


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ImgAug(img, boxes, data_aug_seq):
    """img augumentation
    Args:
        img: ndarray img with any shape [h, w, c]
        corner_bbox: a list or ndarray of bbox with shape [n, 4],
                     encoded by [xmin, ymin, xmax, ymax]
        data_aug_seq: Sequence of augmentations
    Return:
        img: after augumentation
        corner_bbox: after augumentation
    """
    boxes = np.array(boxes)
    boxes = xywh2xyxy_np(boxes)
    bounding_boxes = BoundingBoxesOnImage([BoundingBox(*box) for box in boxes], shape=img.shape)

    image_aug, bounding_boxes = data_aug_seq(image=img, bounding_boxes=bounding_boxes)
    bounding_boxes = bounding_boxes.clip_out_of_image()

    # Convert bounding boxes back to numpy
    boxes = np.zeros((len(bounding_boxes), 4))
    for box_idx, box in enumerate(bounding_boxes):
        # Extract coordinates for unpadded + unscaled image
        x1 = box.x1
        y1 = box.y1
        x2 = box.x2
        y2 = box.y2

        # Returns (cx, cy, w, h)
        boxes[box_idx, 0] = ((x1 + x2) / 2)
        boxes[box_idx, 1] = ((y1 + y2) / 2)
        boxes[box_idx, 2] = (x2 - x1)
        boxes[box_idx, 3] = (y2 - y1)

    return boxes, image_aug


def normalize_data(raw_img, corner_bbox=None, size=416):
    """
    make the raw imgs and raw labels into a standard scalar.
    Args:
        raw_imgs: img with any height and width
        corner_bboxes: label encoded by [ymin, xmin, ymax, xmax]
        size: the output img size, default is (224,224)---(height, width)
    Return:
        norm_imgs: a list of img with the same height and width, and its pixel
                    value is between [-1., 1.]
        norm_corner_bboxes: a list of corner_bboxes [xmin, ymin, xmax, ymax],
                    and its value is between [0., 1.]
    """
    height, width, channels = raw_img.shape
    if corner_bbox is not None:
        norm_xmin = corner_bbox[:, 0] / width
        norm_ymin = corner_bbox[:, 1] / height
        norm_xmax = corner_bbox[:, 2] / width
        norm_ymax = corner_bbox[:, 3] / height
        norm_corner_bbox = np.stack([norm_xmin, norm_ymin, norm_xmax, norm_ymax], axis=-1)
        img = cv2.resize(raw_img, dsize=(size, size))
        height, width, channels = img.shape
        img = img.reshape(channels, height, width)
        # img = (2.0 / 255.0) * img - 1.0
        img = img / 255.0

        return img, norm_corner_bbox
    else:
        img = cv2.resize(raw_img, dsize=(size, size))
        height, width, channels = img.shape
        img = img.reshape(channels, height, width)
        # img = (2.0 / 255.0) * img - 1.0
        img = img / 255.0

        return img


if __name__ == "__main__":
    bboxes = np.array(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.2, 0.2, 0.6, 0.6],
            [0.7, 0.7, 0.9, 0.9],
            [0.55, 0.55, 0.8, 0.8],
        ]
    )
    bbox_corners = box_np_ops.minmax_to_corner_2d(bboxes)
    print(bbox_corners.shape)

