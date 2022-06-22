import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    box1 = box1.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class IoULoss(nn.Module):
    """
    Regression loss for an output tensor
      Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, dim)
        mask: (batch, num_anchors, grid_size, grid_size)
    """

    def __init__(self, GIoU=False, DIoU=False, CIoU=False, reduction='sum'):
        super(IoULoss, self).__init__()
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.reduction = reduction

    def forward(self, pred, target):
        ious = bbox_iou(pred, target, x1y1x2y2=False, GIoU=self.GIoU, DIoU=self.DIoU, CIoU=self.CIoU)

        if self.reduction == 'sum':
            return (1.0 - ious).sum(), ious
        else:
            return (1.0 - ious).mean(), ious


class BBoxRegLoss(nn.Module):
    """
    Regression loss for an output tensor
      Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, dim)
        mask: (batch, num_anchors, grid_size, grid_size)
    """

    def __init__(self, reduction='sum'):
        super(BBoxRegLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask):
        pos_num = mask.sum()
        pred_pos = pred[mask]
        target_pos = target[mask]
        loss = F.smooth_l1_loss(pred_pos, target_pos, reduction='sum')

        if pos_num == 0:
            return 0.0
        else:
            if self.reduction == 'sum':
                return loss
            else:
                return loss / pos_num


class GridRegLoss(nn.Module):
    """
    Regression loss for an output tensor
      Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, dim)
        mask: (batch x num_anchors x grid_size x grid_size)
    """

    def __init__(self, reduction='sum'):
        super(GridRegLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pos_mask = target > 0
        neg_mask = target == 0
        pos_num = pos_mask.sum()
        pos_loss = F.mse_loss(pred[pos_mask], target[pos_mask], reduction='sum')
        neg_loss = F.mse_loss(pred[neg_mask], target[neg_mask], reduction='sum')

        if pos_num == 0:
            return neg_loss
        else:
            if self.reduction == 'sum':
                return pos_loss + neg_loss
            else:
                return (pos_loss + neg_loss) / pos_num


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward(self, pred, target, mask):

        pos_num = mask.sum()
        logpt = F.log_softmax(pred, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target)
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum() / pos_num


class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction='sum'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, pos_mask, hard_neg_mask):
        """
        Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, num_classes)
        mask: (batch x num_anchors x grid_size x grid_size)
        """
        pos_num = pos_mask.sum()
        loss = F.cross_entropy(pred, target, reduction='none', label_smoothing=0.1)
        pos_loss = loss[pos_mask].sum()
        neg_loss = loss[hard_neg_mask].sum()

        if pos_num == 0:
            return neg_loss
        else:
            if self.reduction == 'sum':
                return pos_loss + neg_loss
            else:
                return (pos_loss + neg_loss) / pos_num


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='sum'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, num_classes)
        mask: (batch x num_anchors x grid_size x grid_size)
        """
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)

        return loss
