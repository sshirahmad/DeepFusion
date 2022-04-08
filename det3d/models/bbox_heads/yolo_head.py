import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import math
import time
import datetime
from det3d.torchie.cnn import xavier_init, kaiming_init
from det3d.core import box_torch_ops, box_np_ops
import matplotlib.pyplot as plt
import numpy as np
from ..registry import HEADS
from det3d.models.losses.yolo_loss import BBoxRegLoss, GridRegLoss, CrossEntropyLoss
from ..utils import build_norm_layer


def hard_negative_mining(neg_mask, pos_mask, pred_cls, num_samples, neg_ratio=5.):
    neg_score = torch.where(neg_mask.type(torch.bool), nn.Softmax(dim=-1)(pred_cls)[:, 0], 1 - neg_mask)  # take false positives

    # Number of negative entries to select.
    pos_num = torch.sum(pos_mask)
    max_neg_num = torch.sum(neg_mask)
    max_neg_num = max_neg_num.type(torch.int32)
    if pos_num != 0:
        n_neg = (neg_ratio * pos_num).type(torch.int32)
        n_neg = torch.minimum(n_neg, max_neg_num)

        val, idxes = torch.topk(-neg_score, k=n_neg)  # take negatives with lowest score (hard negatives)
        max_value = -val[-1]

        hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        hard_neg_mask[idxes] = True
    else:
        hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        max_value = 0

    return hard_neg_mask, max_value


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor=None, size=None, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, x):
        if self.size is not None:
            x = F.interpolate(x, size=self.size, mode=self.mode)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def batch_norm(num_features, eps=1e-05, momentum=0.1):
    norm_cfg = dict(type="BN", eps=eps, momentum=momentum)
    return build_norm_layer(norm_cfg, num_features)[1]


def conv2d(inp, oup, kernel, stride, bias=False):
    pad = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, pad, bias=bias),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(0.2, inplace=True),
    )


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.Unfold(kernel_size=block_size, stride=block_size)(x)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


@HEADS.register_module
class YOLOHead(nn.Module):
    """Detection layer"""

    def __init__(self, anchors,
                 tasks,
                 img_dim,
                 noobj_scale,
                 obj_scale,
                 zero_scale,
                 value_scale,
                 feature_map_channels,
                 grid_maps,
                 attention_channels,
                 first_grid_channels,
                 second_grid_channels,
                 third_grid_channels,
                 num_classes,
                 last_conv_channels,
                 logger=None):
        super(YOLOHead, self).__init__()
        self.class_names = [t["class_names"] for t in tasks]
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.sigma_class = nn.Parameter(torch.tensor([0.0]))
        self.sigma_loc = nn.Parameter(torch.tensor([0.0]))
        self.grid_size1 = grid_maps[0]
        self.grid_size2 = grid_maps[1]
        self.grid_size3 = grid_maps[2]

        self.class_loss = CrossEntropyLoss()
        self.bbox_loss = BBoxRegLoss()

        self.last_conv_channels = last_conv_channels

        # 2D convolutions of weight combination layer
        self.wcl_conv2d = nn.ModuleList()
        for channel in feature_map_channels:
            self.wcl_conv2d.append(conv2d(channel, channel // 2, 1, 1))

        # attention modules of weight combination layer
        self.attention_module = nn.ModuleList()
        for channel in attention_channels:
            self.attention_module.append(SELayer(channel))

        # last convolutional layers
        layers = []
        for i in range(1, len(last_conv_channels)):
            if i % 2 == 1:
                layers.append(conv2d(last_conv_channels[i - 1], last_conv_channels[i], 1, 1, bias=False))
            else:
                layers.append(conv2d(last_conv_channels[i - 1], last_conv_channels[i], 3, 1, bias=False))
        layers.append(batch_norm(last_conv_channels[-1]))
        self.last_conv2d = nn.Sequential(*layers)

        self._initialize_weights()

        if not logger:
            logger = logging.getLogger("YOLOHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

    def _initialize_weights(self, init_bias=-2.19):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
                # xavier_init(m)
                # m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _compute_grid_offsets(self, grid_size, cuda=True):
        g = grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        stride = self.img_dim
        # Calculate offsets for each grid
        grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        center_x = (grid_x + 0.5) / grid_size
        grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        center_y = (grid_y + 0.5) / grid_size
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        return center_x, center_y, anchor_w, anchor_h, stride

    def forward(self, multi_scale_features):
        features_names = list(multi_scale_features.keys())
        h = []
        for i in range(len(features_names)):
            feature_map = multi_scale_features[features_names[i]]
            h.append(feature_map.shape[-1])

        # Weight Combination Layer
        feats = []
        for i in range(len(features_names)):
            feature_map = multi_scale_features[features_names[i]]

            # repeat_map = grid_maps[i].unsqueeze(1).repeat(1, feature_map.shape[1], 1, 1)
            # feature_map = torch.mul(feature_map, repeat_map)

            feature_map = self.wcl_conv2d[i](feature_map)

            if i < len(features_names) - 1:
                feature_map = space_to_depth(feature_map, block_size=h[i] // h[-1])
                # feature_map = self.downsample_feature(feature_map)
            feature_map = self.attention_module[i](feature_map)
            feats.append(feature_map)

        net = torch.cat(feats, dim=1)

        # Add few convolutional layers to YOLO
        net = self.last_conv2d(net)

        grid_size = net.size(2)
        num_samples = net.size(0)

        yolo_prediction = (
            net.view(num_samples, self.num_anchors, 6, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # Get yolo outputs
        x = yolo_prediction[..., 0]  # Center x
        y = yolo_prediction[..., 1]  # Center y
        w = yolo_prediction[..., 2]  # Width
        h = yolo_prediction[..., 3]  # Height
        pred_cls = yolo_prediction[..., 4:]  # Cls pred.

        pred_dicts = {
            "pred_boxes": torch.stack((x, y, w, h), dim=-1),
            "pred_cls": pred_cls
        }

        return pred_dicts

    def loss(self, example, pred_dicts, **kwargs):
        """
        Compute YOLO loss
        """

        tboxes = example['gt_boxes_cls'][..., :-1]
        tcls = example['gt_boxes_cls'][..., -1].view(-1).long()

        pred_boxes = pred_dicts["pred_boxes"]
        pred_cls = pred_dicts["pred_cls"]

        batch_size = pred_cls.shape[0]
        pred_cls = pred_cls.view(-1, 2)

        obj_mask = example["obj_mask"]
        pos_mask = obj_mask.view(-1)
        neg_mask = example["noobj_mask"].view(-1)
        neg_mask = neg_mask.type(torch.float32)

        "visualize grid maps"
        # pred_map1_vis = pmap1[0].cpu().detach().numpy()
        # pred_map2_vis = pmap2[0].cpu().detach().numpy()
        # pred_map3_vis = pmap3[0].cpu().detach().numpy()
        # true_map1_vis = tmap1[0].cpu().detach().numpy()
        # true_map2_vis = tmap2[0].cpu().detach().numpy()
        # true_map3_vis = tmap3[0].cpu().detach().numpy()
        # pred_map1_vis = cv2.resize(pred_map1_vis, (self.img_dim, self.img_dim))
        # pred_map2_vis = cv2.resize(pred_map2_vis, (self.img_dim, self.img_dim))
        # pred_map3_vis = cv2.resize(pred_map3_vis, (self.img_dim, self.img_dim))
        # true_map1_vis = cv2.resize(true_map1_vis, (self.img_dim, self.img_dim))
        # true_map2_vis = cv2.resize(true_map2_vis, (self.img_dim, self.img_dim))
        # true_map3_vis = cv2.resize(true_map3_vis, (self.img_dim, self.img_dim))
        # cv2.imshow("predicted_grid_map1", pred_map1_vis)
        # cv2.imshow("predicted_grid_map2", pred_map2_vis)
        # cv2.imshow("predicted_grid_map3", pred_map3_vis)
        # cv2.imshow("true_grid_map1", true_map1_vis)
        # cv2.imshow("true_grid_map2", true_map2_vis)
        # cv2.imshow("true_grid_map3", true_map3_vis)
        # cv2.waitKey(0)

        # Hard negative mining
        hard_neg_mask, max_value = hard_negative_mining(neg_mask, pos_mask, pred_cls, batch_size, neg_ratio=3.)

        # Regression loss for dimension and offset
        sum_loc_loss = self.bbox_loss(pred_boxes, tboxes, obj_mask)
        loc_loss = 0.5 * (torch.exp(-self.sigma_loc) * sum_loc_loss + self.sigma_loc)

        # classification loss
        sum_cls_loss = self.class_loss(pred_cls, tcls, pos_mask, hard_neg_mask, batch_size)
        cls_loss = 1. * (torch.exp(-self.sigma_class) * sum_cls_loss + self.sigma_class)

        loss = loc_loss + cls_loss

        pos_num = pos_mask.sum()

        ret = {}
        ret.update({'loss': loss, 'cls_loss': sum_cls_loss, 'loc_loss': sum_loc_loss,
                    'sigma_class': self.sigma_class, 'sigma_loc': self.sigma_loc,
                    'num_pedestrians': pos_num, 'max_hard_value': max_value})

        return ret

    @torch.no_grad()
    def predict(self, example, pred_dicts, test_cfg, **kwargs):
        """
        decode, nms, then return the detection result.
        """
        # get loss info
        prediction = pred_dicts["pred_boxes"]
        x = prediction[..., 0]
        y = prediction[..., 1]
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_cls = pred_dicts["pred_cls"]

        batch_size = prediction.shape[0]

        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]



        # Add offset and scale with anchors
        grid_x, grid_y, anchor_w, anchor_h, stride = self._compute_grid_offsets(grid_size=self.grid_size3,
                                                                                     cuda=prediction.is_cuda)
        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x * anchor_w + grid_x
        pred_boxes[..., 1] = y * anchor_h + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        pred_cls = torch.nn.Softmax(dim=-1)(pred_cls)

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * stride,
                pred_cls.view(batch_size, -1, 2),
            ),
            -1,
        )

        rets = []
        if test_cfg.get('per_class_nms', False):
            pass
        else:
            rets = self.post_processing(output, test_cfg)

        # Merge batches results
        ret_list = []
        for i in range(batch_size):
            ret = {}
            if rets[i] is not None:
                ret = rets[i]
                ret['metadata'] = meta_list[i]
            else:
                ret['boxes'] = None
                ret['scores'] = None
                ret['metadata'] = meta_list[i]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def post_processing(self, predictions, test_cfg):
        """
            Removes detections with lower object confidence score than 'conf_thres' and performs
            Non-Maximum Suppression to further filter detections.
            Returns detections with shape: (x1, y1, x2, y2, object_conf)
        """
        conf_thres = test_cfg.conf_thres
        nms_thres = test_cfg.nms_thres
        keep_top_k = test_cfg.keep_top_k

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        predictions[..., :4] = box_np_ops.xywh2xyxy(predictions[..., :4])

        prediction_dicts = [None for _ in range(len(predictions))]
        for image_i, image_pred in enumerate(predictions):
            yolo_scores = image_pred[..., -1].view(-1)
            yolo_boxes = image_pred[..., :4].view(-1, 4)

            # Filter out confidence scores below threshold
            mask = yolo_scores > conf_thres
            scores = yolo_scores[mask]
            bboxes = yolo_boxes[mask]

            # If none are remaining => process next image
            if not bboxes.size(0):
                continue

            # Sort the boxes in decreasing confidence score
            sorted_scores, indices = torch.sort(scores, 0, descending=True)
            sorted_boxes = bboxes[indices]

            if sorted_boxes.size(0) > keep_top_k:
                sorted_boxes = sorted_boxes[:keep_top_k, :]
                sorted_scores = sorted_scores[:keep_top_k]

            # Perform non-maximum suppression
            keep_idx = torchvision.ops.nms(sorted_boxes, sorted_scores, nms_thres)
            if keep_idx is not None:
                prediction_dict = {
                    'boxes': bboxes[keep_idx].view(-1, 4),
                    'scores': scores[keep_idx]
                }
                prediction_dicts[image_i] = prediction_dict

        return prediction_dicts
