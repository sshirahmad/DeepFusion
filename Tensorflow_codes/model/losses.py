import tensorflow.compat.v1 as tf
import math


class BoxRegLoss:
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, bbox_pred, bbox_gt, pos_mask):
        """
        Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        """
        x = tf.reshape((bbox_pred - bbox_gt), [-1, 4]) * tf.expand_dims(pos_mask, axis=-1)
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        loss = 0.5 * ((absx - 1) * minx + absx)  # smooth_l1
        loss = tf.reduce_sum(loss)
        if self.reduction == 'sum':
            return loss
        else:
            return tf.cond(
                tf.equal(tf.reduce_sum(pos_mask), 0.0),
                true_fn=lambda: loss,
                false_fn=lambda: loss / tf.reduce_sum(pos_mask),
            )


class CIoULoss:
    """
    Regression loss for an output tensor
      Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, dim)
        mask: (batch, num_anchors, grid_size, grid_size)
    """

    def __init__(self, GIoU=False, DIoU=False, CIoU=False, reduction='sum'):

        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.reduction = reduction

    def __call__(self, bbox_pred, bbox_gt, pos_mask):
        pred_pos = tf.reshape(bbox_pred, [-1, 4]) * tf.expand_dims(pos_mask, axis=-1)
        target_pos = tf.reshape(bbox_gt, [-1, 4]) * tf.expand_dims(pos_mask, axis=-1)

        b1_x1, b1_x2 = pred_pos[:, 1] - pred_pos[:, 3] / 2, pred_pos[:, 1] + pred_pos[:, 3] / 2
        b1_y1, b1_y2 = pred_pos[:, 0] - pred_pos[:, 2] / 2, pred_pos[:, 0] + pred_pos[:, 2] / 2
        b2_x1, b2_x2 = target_pos[:, 1] - target_pos[:, 3] / 2, target_pos[:, 1] + target_pos[:, 3] / 2
        b2_y1, b2_y2 = target_pos[:, 0] - target_pos[:, 2] / 2, target_pos[:, 0] + target_pos[:, 2] / 2

        # Intersection area
        inter_rect_x1 = tf.maximum(b1_x1, b2_x1)
        inter_rect_y1 = tf.maximum(b1_y1, b2_y1)
        inter_rect_x2 = tf.minimum(b1_x2, b2_x2)
        inter_rect_y2 = tf.minimum(b1_y2, b2_y2)
        inter = tf.clip_by_value(inter_rect_x2 - inter_rect_x1 + 1, 0, 2) * tf.clip_by_value(
            inter_rect_y2 - inter_rect_y1 + 1, 0, 2)

        # Union Area
        w1, h1 = b1_x2 - b1_x1 + 1, b1_y2 - b1_y1 + 1
        w2, h2 = b2_x2 - b2_x1 + 1, b2_y2 - b2_y1 + 1
        union = w1 * h1 + w2 * h2 - inter + 1e-16

        iou = inter / union  # iou

        if self.GIoU or self.DIoU or self.CIoU:
            cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)  # convex height
            if self.GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + 1e-16  # convex area
                cious = iou - (c_area - union) / c_area  # GIoU
            if self.DIoU or self.CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                # convex diagonal squared
                c2 = cw ** 2 + ch ** 2 + 1e-16
                # centerpoint distance squared
                rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
                if self.DIoU:
                    cious = iou - rho2 / c2  # DIoU
                elif self.CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
                    alpha = v / (1 - iou + v)
                    cious = iou - (rho2 / c2 + v * alpha)  # CIoU

        loss = tf.reduce_sum(1.0 - cious)

        if self.reduction == 'sum':
            return loss
        else:
            return loss / tf.reduce_sum(pos_mask)


class GridRegLoss:
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, grid_pred, grid_gt):
        """
        Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        """
        pos = tf.cast(grid_gt > 0., dtype=tf.float32)
        pos_num = tf.reduce_sum(pos)
        x = tf.reshape((grid_pred - grid_gt), [-1])
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        loss = 0.5 * ((absx - 1) * minx + absx)  # smooth_l1
        loss = tf.reduce_sum(loss)

        if self.reduction == 'sum':
            return loss
        else:
            return tf.cond(
                tf.equal(pos_num, 0.0),
                true_fn=lambda: loss,
                false_fn=lambda: loss / pos_num,
            )


class CrossEntropyLoss:
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, logits_pred, label_gt, pos_mask, hard_neg_mask):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred, labels=tf.reshape(label_gt, [-1]))

        pos_loss = tf.reduce_sum(loss * pos_mask)
        neg_loss = tf.reduce_sum(loss * hard_neg_mask)

        if self.reduction == 'sum':
            return pos_loss + neg_loss
        else:
            return tf.cond(
                tf.equal(tf.reduce_sum(pos_mask), 0.0),
                true_fn=lambda: pos_loss + neg_loss,
                false_fn=lambda: (pos_loss + neg_loss) / tf.reduce_sum(pos_mask),
            )
