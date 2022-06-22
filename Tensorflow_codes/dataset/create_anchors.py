import random
import argparse
import numpy as np
import cv2
from det3d.core.bbox import box_np_ops


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum / n


def print_anchors(centroids):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i, 0] * 416)) + ',' + str(int(anchors[i, 1] * 416)) + ', '

    print(out_string[:-2])


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.full(ann_num, -1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_num) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)  # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances - distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def generate_anchors(db_infos, num_anchors, img_size, class_name="pedestrian"):
    # run k_mean to find the anchors
    annotation_dims = np.ndarray(shape=(1, 2))
    for info in db_infos:
        filename = info['cam_front_path']
        image = cv2.imread(filename)
        height, width, depth = image.shape

        gt_names = info["gt_names"]
        gt_boxes_mask = np.array(
            [n in class_name for n in gt_names], dtype=np.bool_
        )

        boxes = info["gt_boxes"][gt_boxes_mask].astype(np.float32)
        intrinsic = info["cam_intrinsic"]
        minmax_boxes = box_np_ops.convert_box3d_box2d(boxes, intrinsic)
        minmax_boxes[np.isnan(minmax_boxes)] = 0

        if len(minmax_boxes != 0):
            relative_w = (minmax_boxes[:, 2] - minmax_boxes[:, 0]) / width * img_size
            relative_h = (minmax_boxes[:, 3] - minmax_boxes[:, 1]) / height * img_size
            wh = np.stack((relative_w, relative_h), axis=-1)
            annotation_dims = np.concatenate((annotation_dims, wh), axis=0)
    annotation_dims = annotation_dims[1:]
    centroids = run_kmeans(annotation_dims, num_anchors)
    area = []
    for c in centroids:
        w, h = c
        area.append(w * h)

    idx = np.argsort(area)
    sorted_centroids = centroids[idx]

    return sorted_centroids


def generate_anchors_caltech(db_infos, num_anchors, img_size, class_name="pedestrian"):
    # run k_mean to find the anchors
    annotation_dims = np.ndarray(shape=(1, 2))
    for info in db_infos:
        filename = info['image_path']
        image = cv2.imread(filename)
        height, width, depth = image.shape

        gt_names = info["gt_names"]
        gt_boxes_mask = np.array(
            [n in class_name for n in gt_names], dtype=np.bool_
        )
        gt_boxes = info["gt_boxes"][gt_boxes_mask].astype(np.float32)
        gt_boxes[np.isnan(gt_boxes)] = 0

        if len(gt_boxes != 0):
            relative_w = gt_boxes[:, 2] / width * img_size
            relative_h = gt_boxes[:, 3] / height * img_size
            wh = np.stack((relative_w, relative_h), axis=-1)
            annotation_dims = np.concatenate((annotation_dims, wh), axis=0)
    annotation_dims = annotation_dims[1:]
    centroids = run_kmeans(annotation_dims, num_anchors)
    area = []
    for c in centroids:
        w, h = c
        area.append(w*h)
    
    idx = np.argsort(area)
    sorted_centroids = centroids[idx]

    return sorted_centroids
