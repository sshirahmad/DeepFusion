import tensorflow.compat.v1 as tf
import math
import numpy as np

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception:
    raise ImportError("Pls install imgaug")

ia.seed(1)

# a seq of img augumentation
data_aug_seq = iaa.SomeOf(3, [
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.2)),  # random crops

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=[-25, 25]
    )
], random_order=True)  # apply augmenters in random order


def imgaugboxes_2_corner_bboxes(imgaugboxes):
    """"""
    bboxes = []
    for bbox in imgaugboxes.bounding_boxes:
        bboxes.append(np.array([bbox.y1, bbox.x1, bbox.y2, bbox.x2]))

    return np.array(bboxes)


def img_aug(img, corner_bbox):
    """img augumentation
    Args:
        img: ndarray img with any shape [h, w, c]
        corner_bbox: a list or ndarray of bbox with shape [n, 4],
                     encoded by [ymin, xmin, ymax, xmax]
    Return:
        img: after augumentation
        cornet_bbox: after augumentation
    """

    bboxes = []
    for bbox in corner_bbox:
        x1 = bbox[1]
        y1 = bbox[0]
        x2 = bbox[3]
        y2 = bbox[2]
        bboxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label="person"))

    bbs = ia.BoundingBoxesOnImage(bboxes, shape=img.shape)

    seq_det = data_aug_seq.to_deterministic()

    ## augumentation ##
    image_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().clip_out_of_image()

    bboxes = imgaugboxes_2_corner_bboxes(bbs_aug)

    return image_aug, bboxes


def fuse_scores(predictions_layer, localizations_layer, grid_map, img_size, batch_size):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
        :param localizations_layer:
        :param predictions_layer:
        :param select_threshold:
        :param scope:
        :param ignore_class:
        :param num_classes:
    """
    with tf.name_scope('scores_fuse_layer'):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tf.shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer, [p_shape[0], -1, p_shape[-1]])
        l_shape = tf.shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer, [l_shape[0], -1, l_shape[-1]])
        g_shape = tf.shape(localizations_layer)
        grid_map = tf.reshape(grid_map, [g_shape[0], img_size // 8, img_size // 8])

        f_scores = []
        for i in range(batch_size):
            grid = grid_map[i, :, :]
            bboxes = localizations_layer[i, :, :]
            scores = predictions_layer[i, :, :]
            num_boxes = ((img_size // 8)**2 + (img_size // 16)**2 + (img_size // 32)**2) * 3
            for j in range(num_boxes):
                ymin = tf.cast(bboxes[j, 0] * img_size // 8, dtype=tf.int32)
                xmin = tf.cast(bboxes[j, 1] * img_size // 8, dtype=tf.int32)
                ymax = tf.cast(bboxes[j, 2] * img_size // 8, dtype=tf.int32)
                xmax = tf.cast(bboxes[j, 3] * img_size // 8, dtype=tf.int32)
                grid_score = tf.reduce_mean(grid[ymin:(ymax + 1), xmin:(xmax + 1)])
                f_scores.append([scores[j, 0], tf.sqrt(grid_score * scores[j, -1])])

        f_scores = tf.reshape(f_scores, [p_shape[0], -1, p_shape[-1]])

        return f_scores


def bboxes_select(predictions_layer, localizations_layer,
                  select_threshold=None,
                  num_classes=2,
                  ignore_class=0,
                  scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
        :param localizations_layer:
        :param predictions_layer:
        :param select_threshold:
        :param scope:
        :param ignore_class:
        :param num_classes:
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tf.shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tf.shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def bboxes_sort(scores, bboxes, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c])
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        indices = tf.argsort(scores, axis=-1, direction='DESCENDING')
        scores = tf.gather(scores, indices, axis=1, batch_dims=1)
        bboxes = tf.gather(bboxes, indices, axis=1, batch_dims=1)

        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)

        return scores, bboxes


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_y1, b1_x1, b1_y2, b1_x2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_y1, b2_x1, b2_y2, b2_x2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip((inter_rect_x2 - inter_rect_x1 + 1), 0, None) * np.clip((inter_rect_y2 - inter_rect_y1 + 1), 0,
                                                                                 None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


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
    if len(prec) == 0:
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


def compute_ap(tp, num_labels):
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

    # Create Precision-Recall curve and compute AP for each class
    n_gt = num_labels  # Number of ground truth objects
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


def match_boxes(pred_boxes, pred_scores, target_boxes, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """

    true_positives = np.zeros(pred_boxes.shape[0])
    detected_boxes = []
    matched_boxes = []
    matched_scores = []
    for pred_i, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):

        # If targets are found break
        if len(detected_boxes) == len(target_boxes):
            break

        ious = bbox_iou(pred_box[np.newaxis, ...], target_boxes)
        iou = np.max(ious)
        box_index = np.argmax(ious)
        if iou >= iou_threshold and box_index not in detected_boxes:
            true_positives[pred_i] = 1
            detected_boxes += [box_index]
            matched_boxes += [pred_box]
            matched_scores += [pred_score]

    return true_positives, matched_boxes, matched_scores


if __name__ == '__main__':
    import cv2

    img = np.ones(shape=(224, 224, 3), dtype=np.uint8) * 255
    bbox = np.array([[10, 10, 50, 50], [30, 40, 150, 150]])
    score = np.array([0.8, 0.3])
    label = np.array([1, 2])
    category_index = {0: {"name": "background"},
                      1: {"name": "person"},
                      2: {"name": "vehicle"}}
    img = visualize_boxes_and_labels_on_image_array(img, bbox, label, score, category_index)
    # img = draw_bounding_box_on_image_array(img, 10, 10, 150, 150, color='LightCoral',thickness=2,display_str_list=["person"])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass
