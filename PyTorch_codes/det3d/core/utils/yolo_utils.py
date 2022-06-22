from __future__ import division
import math
import time
import tqdm
import cv2
import torch
import numpy as np
from det3d.core.bbox.box_np_ops import overlap_jit



def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def log_average_miss_rate(prec, rec):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def compute_ap(tp, pred_scores, labels):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-pred_scores)
    tp, conf = tp[i], pred_scores[i]
    # Create Precision-Recall curve and compute AP for each class
    n_gt = len(labels)  # Number of ground truth objects
    n_p = len(tp)  # Number of predicted objects

    if n_p == 0 or n_gt == 0:
        ap = 0
        r = 0
        p = 0
        precision_curve = 0
        recall_curve = 0
    else:
        # Accumulate FPs and TPs
        fpc = (1 - tp).cumsum()
        tpc = tp.cumsum()

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        r = recall_curve[-1]

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p = precision_curve[-1]

        # AP from recall-precision curve
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], precision_curve, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return precision_curve, recall_curve, p, r, ap, f1


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    batch_matched_inds = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]

        true_positives = np.zeros(pred_boxes.shape[0])
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        detected_boxes = []
        matched_pred_boxes = []
        if len(annotations):
            target_boxes = annotations

            for pred_i, pred_box in enumerate(pred_boxes):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Find the best matching target for our predicted box
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                # Check if the iou is above the min threshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
                    matched_pred_boxes += [pred_i]

        batch_metrics.append([true_positives, pred_scores])
        batch_matched_inds.append(matched_pred_boxes)

    return batch_metrics, batch_matched_inds


def bbox_wh_iou(wh1, wh2):
    w1, h1 = wh1[2] - wh1[0], wh1[3] - wh1[1]
    w2, h2 = wh2[:, 0], wh2[:, 1]
    inter_area = np.minimum(w1, w2) * np.minimum(h1, h2)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(gt_boxes, img_size, anchors, surrounding_size, top_k, obj_thres):
    corner_bboxes = xywh2xyxy_np(gt_boxes[:, 1:])

    nA = len(anchors)
    nG = list([img_size // 8, img_size // 16, img_size // 32])
    valid_anchors = list(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    best_anchor_ind = [np.argmax(bbox_wh_iou(gt_box, anchors)) for gt_box in corner_bboxes]

    # Output tensors
    classifier_maps = []
    yolo_maps = []
    obj_masks = []
    noobj_masks = []
    for k, size in enumerate(nG):
        # make target boxes relative to grid map
        target_boxes = corner_bboxes

        obj_mask = np.zeros(shape=(nA // 3, size, size), dtype=np.bool_)
        noobj_mask = np.ones(shape=(nA // 3, size, size), dtype=np.bool_)
        tx = np.zeros(shape=(nA // 3, size, size), dtype=np.float32)
        ty = np.zeros(shape=(nA // 3, size, size), dtype=np.float32)
        tw = np.zeros(shape=(nA // 3, size, size), dtype=np.float32)
        th = np.zeros(shape=(nA // 3, size, size), dtype=np.float32)
        grid_maps = np.zeros(shape=(size, size), dtype=np.float32)

        # create groundtruth of YOLO classifiers
        h_per_cell = 1 / size
        w_per_cell = 1 / size
        center_location_h_index = np.int32((target_boxes[:, 1] + target_boxes[:, 3]) / (2 * h_per_cell))
        center_location_w_index = np.int32((target_boxes[:, 0] + target_boxes[:, 2]) / (2 * w_per_cell))

        scaled_anchors = anchors[valid_anchors[k]] / img_size

        # find the index of surrounding priori boxes
        total_ovr_info = []
        priori_box_index = []
        for iter in range(len(center_location_w_index)):
            if center_location_h_index[iter] - surrounding_size >= 0:
                min_h_index = center_location_h_index[iter] - surrounding_size
            else:
                min_h_index = 0
            if center_location_h_index[iter] + surrounding_size <= size - 1:
                max_h_index = center_location_h_index[iter] + surrounding_size
            else:
                max_h_index = size - 1
            if center_location_w_index[iter] - surrounding_size >= 0:
                min_w_index = center_location_w_index[iter] - surrounding_size
            else:
                min_w_index = 0
            if center_location_w_index[iter] + surrounding_size <= size - 1:
                max_w_index = center_location_w_index[iter] + surrounding_size
            else:
                max_w_index = size - 1

            h_indexes = np.arange(min_h_index, max_h_index + 1, 1)
            w_indexes = np.arange(min_w_index, max_w_index + 1, 1)

            # compute the IOU of each bounding box with its surrounding priori boxes
            ovr_info = []
            for wIndex in w_indexes:
                for hIndex in h_indexes:
                    # calculate center and dimensions of surrounding anchor boxes
                    y_c = (hIndex + 0.5) * h_per_cell
                    x_c = (wIndex + 0.5) * w_per_cell
                    h_p = scaled_anchors[:, 1]
                    w_p = scaled_anchors[:, 0]

                    # calculate IoU of surrounding anchor boxes and groundtruths
                    for i in range(len(scaled_anchors)):
                        x_min = x_c - w_p[i] / 2
                        x_max = x_c + w_p[i] / 2
                        y_min = y_c - h_p[i] / 2
                        y_max = y_c + h_p[i] / 2

                        areaP = (x_max - x_min) * (y_max - y_min)
                        areaG = (target_boxes[iter, 2] - target_boxes[iter, 0]) * (
                                target_boxes[iter, 3] - target_boxes[iter, 1])
                        xx1 = np.maximum(x_min, target_boxes[iter, 0])
                        yy1 = np.maximum(y_min, target_boxes[iter, 1])
                        xx2 = np.minimum(x_max, target_boxes[iter, 2])
                        yy2 = np.minimum(y_max, target_boxes[iter, 3])
                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        ovr = inter / (areaP + areaG - inter)

                        ovr_info.append([hIndex, wIndex, i, ovr])

            ovr_info = np.array(ovr_info)
            ovr_info = ovr_info.reshape(-1, 4)
            total_ovr_info.append(ovr_info)

            try:
                inds = np.argsort(ovr_info[:, 3])[::-1]
            except IndexError:
                pass

            num = 0
            for index in inds:
                info = [ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]]
                if info not in priori_box_index:  # to avoid same priorBox match multi groundTruth
                    if num >= top_k:  # this value means we choose the top-k ovr box to be the positive box
                        break
                    priori_box_index.append([ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]])
                    num += 1

        priori_box_index = np.array(priori_box_index, dtype=np.int32)
        priori_box_index = priori_box_index.reshape([-1, top_k, 3])
        mask_grid = np.zeros(len(target_boxes), dtype=np.bool_)
        for ground_truth_index in range(len(total_ovr_info)):
            target_bbox = target_boxes[ground_truth_index]
            if best_anchor_ind[ground_truth_index] in valid_anchors[k]:
                mask_grid[ground_truth_index] = 1
                for index_info in priori_box_index[ground_truth_index]:
                    anchor_ind = int(index_info[2])
                    y_ind = int(index_info[0])
                    x_ind = int(index_info[1])

                    y_c_anchor = (index_info[0] + 0.5) * h_per_cell
                    x_c_anchor = (index_info[1] + 0.5) * w_per_cell
                    h_anchor = scaled_anchors[anchor_ind, 1]
                    w_anchor = scaled_anchors[anchor_ind, 0]

                    y_c_gt = (target_bbox[3] + target_bbox[1]) / 2
                    x_c_gt = (target_bbox[0] + target_bbox[2]) / 2
                    h_gt = target_bbox[3] - target_bbox[1]
                    w_gt = target_bbox[2] - target_bbox[0]

                    # difference between anchor boxes and groundtruth
                    y_t = (y_c_gt - y_c_anchor) / h_anchor
                    x_t = (x_c_gt - x_c_anchor) / w_anchor
                    h_t = np.log(h_gt / h_anchor + 1e-16)
                    w_t = np.log(w_gt / w_anchor + 1e-16)

                    tx[anchor_ind, y_ind, x_ind] = x_t
                    ty[anchor_ind, y_ind, x_ind] = y_t
                    tw[anchor_ind, y_ind, x_ind] = w_t
                    th[anchor_ind, y_ind, x_ind] = h_t

                    obj_mask[anchor_ind, y_ind, x_ind] = 1
                    noobj_mask[anchor_ind, y_ind, x_ind] = 0

            for index_info in total_ovr_info[ground_truth_index]:
                if index_info[-1] >= obj_thres:
                    mask_grid[ground_truth_index] = 1
                    anchor_ind = int(index_info[2])
                    y_ind = int(index_info[0])
                    x_ind = int(index_info[1])

                    y_c_anchor = (index_info[0] + 0.5) * h_per_cell
                    x_c_anchor = (index_info[1] + 0.5) * w_per_cell
                    h_anchor = scaled_anchors[anchor_ind, 1]
                    w_anchor = scaled_anchors[anchor_ind, 0]

                    y_c_gt = (target_bbox[3] + target_bbox[1]) / 2
                    x_c_gt = (target_bbox[0] + target_bbox[2]) / 2
                    h_gt = target_bbox[3] - target_bbox[1]
                    w_gt = target_bbox[2] - target_bbox[0]

                    # difference between anchor boxes and groundtruth
                    y_t = (y_c_gt - y_c_anchor) / h_anchor
                    x_t = (x_c_gt - x_c_anchor) / w_anchor
                    h_t = np.log(h_gt / h_anchor + 1e-16)
                    w_t = np.log(w_gt / w_anchor + 1e-16)

                    tx[anchor_ind, y_ind, x_ind] = x_t
                    ty[anchor_ind, y_ind, x_ind] = y_t
                    tw[anchor_ind, y_ind, x_ind] = w_t
                    th[anchor_ind, y_ind, x_ind] = h_t

                    obj_mask[anchor_ind, y_ind, x_ind] = 1
                    noobj_mask[anchor_ind, y_ind, x_ind] = 0

        # create groundtruth of grid classifiers
        x = np.arange(size, dtype=np.float32)
        y = x.T
        if mask_grid.any():
            for i in range(size):
                for j in range(size):
                    box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                    iou = overlap_jit(target_boxes[mask_grid] * size, box)
                    grid_maps[j, i] = iou.max(0)

        tcls = obj_mask.astype(np.float)
        gt_boxes_cls = np.stack((tx, ty, tw, th, tcls), axis=-1)
        classifier_maps.append(grid_maps)
        yolo_maps.append(gt_boxes_cls)
        obj_masks.append(obj_mask)
        noobj_masks.append(noobj_mask)

    example = {}
    example.update({
        'yolo_map1': yolo_maps[0],
        'yolo_map2': yolo_maps[1],
        'yolo_map3': yolo_maps[2],
        'classifier_map1': classifier_maps[0],
        'classifier_map2': classifier_maps[1],
        'classifier_map3': classifier_maps[2],
        'obj_mask1': obj_masks[0],
        'noobj_mask1': noobj_masks[0],
        'obj_mask2': obj_masks[1],
        'noobj_mask2': noobj_masks[1],
        'obj_mask3': obj_masks[2],
        'noobj_mask3': noobj_masks[2],
    })

    return example


def build_grid_targets(gt_boxes, img_size):
    corner_bboxes = xywh2xyxy_np(gt_boxes[:, 1:])

    nG = list([img_size // 8, img_size // 16, img_size // 32])
    classifier_maps = []
    for size in nG:
        grid_maps = np.zeros(shape=(size, size), dtype=np.float32)
        # create groundtruth of grid classifiers
        x = np.arange(size, dtype=np.float32)
        y = x.T
        for i in range(size):
            for j in range(size):
                box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                iou = overlap_jit(corner_bboxes * size, box)
                grid_maps[j, i] = iou.max(0)

        classifier_maps.append(grid_maps)

    example = {}
    example.update({
        'classifier_map1': classifier_maps[0],
        'classifier_map2': classifier_maps[1],
        'classifier_map3': classifier_maps[2],
    })

    return example

