import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from det3d.core.utils.center_utils import _transpose_and_gather_feat


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
        batch_size = mask.size(0)
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


class CIoULoss(nn.Module):
    """
    Regression loss for an output tensor
      Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, dim)
        mask: (batch, num_anchors, grid_size, grid_size)
    """

    def __init__(self, GIoU=False, DIoU=False, CIoU=False, reduction='sum'):
        super(CIoULoss, self).__init__()
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.reduction = reduction

    def forward(self, pred, target, mask):
        batch_size = mask.size(0)
        pos_num = mask.sum()
        pred_pos = pred[mask]
        target_pos = target[mask]

        b1_x1, b1_x2 = pred_pos[:, 0] - pred_pos[:, 2] / 2, pred_pos[:, 0] + pred_pos[:, 2] / 2
        b1_y1, b1_y2 = pred_pos[:, 1] - pred_pos[:, 3] / 2, pred_pos[:, 1] + pred_pos[:, 3] / 2
        b2_x1, b2_x2 = target_pos[:, 0] - target_pos[:, 2] / 2, target_pos[:, 0] + target_pos[:, 2] / 2
        b2_y1, b2_y2 = target_pos[:, 1] - target_pos[:, 3] / 2, target_pos[:, 1] + target_pos[:, 3] / 2

        # Intersection area
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        inter = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                    min=0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1 + 1, b1_y2 - b1_y1 + 1
        w2, h2 = b2_x2 - b2_x1 + 1, b2_y2 - b2_y1 + 1
        union = w1 * h1 + w2 * h2 - inter + 1e-16

        iou = inter / union  # iou

        if self.GIoU or self.DIoU or self.CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
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
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (1 - iou + v)
                    cious = iou - (rho2 / c2 + v * alpha)  # CIoU

        if self.reduction == 'sum':
            return (1.0 - cious).sum()
        else:
            return (1.0 - cious).sum() / pos_num


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
        batch_size = target.size(0)
        pos_mask = target > 0
        neg_mask = target == 0
        pos_num = pos_mask.sum()
        pos_loss = F.smooth_l1_loss(pred[pos_mask], target[pos_mask], reduction='sum')
        neg_loss = F.smooth_l1_loss(pred[neg_mask], target[neg_mask], reduction='sum')

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

    def forward(self, pred, target, pos_mask, hard_neg_mask, batch_size):
        """
        Arguments:
        pred, target: (batch, num_anchors, grid_size, grid_size, num_classes)
        mask: (batch x num_anchors x grid_size x grid_size)
        """
        pos_num = pos_mask.sum()
        loss = F.cross_entropy(pred, target, reduction='none')
        pos_loss = loss[pos_mask].sum()
        neg_loss = loss[hard_neg_mask].sum()

        if pos_num == 0:
            return neg_loss
        else:
            if self.reduction == 'sum':
                return pos_loss + neg_loss
            else:
                return (pos_loss + neg_loss) / pos_num
