import torch
import logging
import torch.nn as nn
import cv2

import torch.nn.functional as F
import torchvision.ops

from det3d.torchie.cnn import xavier_init, kaiming_init
from det3d.core import box_torch_ops, box_np_ops
from det3d.core import xywh2xyxy
from ..registry import HEADS
from det3d.models.losses.yolo_loss import IoULoss, GridRegLoss, BBoxRegLoss, FocalLoss, CrossEntropyLoss
from ..utils import build_norm_layer


def hard_negative_mining(neg_mask, pos_mask, pred_cls, num_samples, neg_ratio=5.):
    neg_score = torch.where(neg_mask.type(torch.bool), nn.Softmax(dim=-1)(pred_cls)[:, 0],
                            1.0 - neg_mask)  # take false positives

    # Number of negative entries to select.
    pos_num = torch.sum(pos_mask)
    max_neg_num = torch.sum(neg_mask)
    max_neg_num = max_neg_num.type(torch.int32)

    n_neg = (neg_ratio * pos_num).type(torch.int32) + num_samples
    n_neg = torch.minimum(n_neg, max_neg_num)

    val, idxes = torch.topk(-neg_score, k=n_neg)  # take negatives with lowest score (hard negatives)
    max_value = -val[-1]

    hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
    hard_neg_mask[idxes] = True

    return hard_neg_mask, max_value


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
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


def batch_norm(num_features, eps=1e-5, momentum=0.1):
    norm_cfg = dict(type="BN", eps=eps, momentum=momentum)
    return build_norm_layer(norm_cfg, num_features)[1]


def conv2d(inp, oup, kernel, stride, bias=False):
    pad = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, pad, bias=bias),
        batch_norm(oup),
        nn.Mish(),
    )


def conv2d_grid(inp, oup, kernel, stride, bias=False):
    pad = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, pad, bias=bias),
        nn.Sigmoid(),
    )


@HEADS.register_module
class YOLOv3Head(nn.Module):
    """Detection layer"""

    def __init__(self, anchors,
                 img_dim,
                 first_map_conv_channels,
                 second_map_conv_channels,
                 third_map_conv_channels,
                 first_grid_channels,
                 second_grid_channels,
                 third_grid_channels,
                 attention_channels,
                 logger=None):
        super(YOLOv3Head, self).__init__()

        self.class_names = ["pedestrian"]
        self.anchors = anchors
        self.num_anchors = len(anchors) // 3
        self.sigma_class = nn.Parameter(torch.tensor([0.0]))
        self.sigma_loc = nn.Parameter(torch.tensor([0.0]))
        self.sigma_grid = nn.Parameter(torch.tensor([0.0]))
        self.num_predictions = first_map_conv_channels[-1] // self.num_anchors
        self.img_dim = img_dim

        self.class_loss = CrossEntropyLoss(reduction='mean')
        self.bbox_loss = BBoxRegLoss(reduction='mean')
        self.grid_loss = GridRegLoss(reduction='mean')

        # attention modules of weight combination layer
        self.attention_module = nn.ModuleList()
        for channel in attention_channels:
            self.attention_module.append(SELayer(channel))

        # first map 2D convolution layers
        self.first_map_conv2d = nn.ModuleList()
        for i in range(1, len(first_map_conv_channels) - 1):
            if i % 2 == 1:
                self.first_map_conv2d.append(
                    conv2d(first_map_conv_channels[i - 1], first_map_conv_channels[i], 1, 1, bias=False))
            else:
                self.first_map_conv2d.append(
                    conv2d(first_map_conv_channels[i - 1], first_map_conv_channels[i], 3, 1, bias=False))
        self.first_map_conv2d.append(nn.Conv2d(first_map_conv_channels[-2], first_map_conv_channels[-1], 1, bias=True))

        # second map 2D convolution layers
        self.second_map_conv2d = nn.ModuleList()
        for i in range(1, len(second_map_conv_channels) - 1):
            if i % 2 == 1:
                self.second_map_conv2d.append(
                    conv2d(second_map_conv_channels[i - 1], second_map_conv_channels[i], 1, 1, bias=False))
            else:
                self.second_map_conv2d.append(
                    conv2d(second_map_conv_channels[i - 1], second_map_conv_channels[i], 3, 1, bias=False))
        self.second_map_conv2d.append(nn.Conv2d(second_map_conv_channels[-2], second_map_conv_channels[-1], 1, bias=True))

        # third map 2D convolution layers
        self.third_map_conv2d = nn.ModuleList()
        for i in range(1, len(third_map_conv_channels) - 1):
            if i % 2 == 1:
                self.third_map_conv2d.append(
                    conv2d(third_map_conv_channels[i - 1], third_map_conv_channels[i], 1, 1, bias=False))
            else:
                self.third_map_conv2d.append(
                    conv2d(third_map_conv_channels[i - 1], third_map_conv_channels[i], 3, 1, bias=False))
        self.third_map_conv2d.append(nn.Conv2d(third_map_conv_channels[-2], third_map_conv_channels[-1], 1, bias=True))

        self.grid_classifiers = nn.ModuleList()
        # first grid classifier
        layers = []
        for i in range(1, len(first_grid_channels) - 1):
            if i % 2 == 1:
                layers.append(
                    conv2d(first_grid_channels[i - 1], first_grid_channels[i], 1, 1, bias=False))
            else:
                layers.append(
                    conv2d(first_grid_channels[i - 1], first_grid_channels[i], 3, 1, bias=False))
        layers.append(conv2d_grid(first_grid_channels[-2], first_grid_channels[-1], 1, 1, bias=True))
        self.grid_classifiers.append(nn.Sequential(*layers))

        # second grid classifier
        layers = []
        for i in range(1, len(second_grid_channels) - 1):
            if i % 2 == 1:
                layers.append(
                    conv2d(second_grid_channels[i - 1], second_grid_channels[i], 1, 1, bias=False))
            else:
                layers.append(
                    conv2d(second_grid_channels[i - 1], second_grid_channels[i], 3, 1, bias=False))
        layers.append(conv2d_grid(second_grid_channels[-2], second_grid_channels[-1], 1, 1, bias=True))
        self.grid_classifiers.append(nn.Sequential(*layers))

        # third grid classifier
        layers = []
        for i in range(1, len(third_grid_channels) - 1):
            if i % 2 == 1:
                layers.append(
                    conv2d(third_grid_channels[i - 1], third_grid_channels[i], 1, 1, bias=False))
            else:
                layers.append(
                    conv2d(third_grid_channels[i - 1], third_grid_channels[i], 3, 1, bias=False))
        layers.append(conv2d_grid(third_grid_channels[-2], third_grid_channels[-1], 1, 1, bias=True))
        self.grid_classifiers.append(nn.Sequential(*layers))

        # Upsampling grids to lower stage feature map size in training
        self.upsample = Upsample(scale_factor=2)

        # Upsampling grid maps to original image size for decision fusion
        self.upsample_grid = Upsample(size=self.img_dim)

        if not logger:
            logger = logging.getLogger("YOLOv3Head")
        self.logger = logger

        self._initialize_weights()

        logger.info("Finish YOLOV3Head Initialization")

    def _compute_grid_offsets(self, grid_size, valid_anchors, cuda=True):
        g = grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        stride = self.img_dim
        # Calculate offsets for each grid
        grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        center_x = (grid_x + 0.5) / grid_size
        grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        center_y = (grid_y + 0.5) / grid_size
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors[valid_anchors]])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        return center_x, center_y, anchor_w, anchor_h, stride

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m, nonlinearity='relu')  # TODO weight initialization could be changed
                xavier_init(m, distribution="normal")
                # m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, multi_scale_features):
        features_names = list(multi_scale_features.keys())

        # Third stage (large pedestrians)
        third_feature = multi_scale_features[features_names[2]]

        # channel-wise attention module
        weighted_third_feature = self.attention_module[2](third_feature)

        # apply grid classifier
        grid_map3 = self.grid_classifiers[2](weighted_third_feature).view(-1,
                                                                          weighted_third_feature.shape[2],
                                                                          weighted_third_feature.shape[3])

        # spatial attention module
        repeated_grid = grid_map3.unsqueeze(1).repeat(1, weighted_third_feature.shape[1], 1, 1)
        third_pred = torch.mul(weighted_third_feature, repeated_grid)
        for i, layer in enumerate(self.third_map_conv2d):
            third_pred = layer(third_pred)
            if i == 4:
                upsampled_third_feat = self.upsample(third_pred)

        # Second stage (medium pedestrians)
        second_feature = torch.cat((multi_scale_features[features_names[1]], upsampled_third_feat), dim=1)

        # channel-wise attention module
        weighted_second_feature = self.attention_module[1](second_feature)

        # apply grid classifier
        grid_map2 = self.grid_classifiers[1](weighted_second_feature).view(-1,
                                                                           weighted_second_feature.shape[2],
                                                                           weighted_second_feature.shape[3])

        # spatial attention module
        repeated_grid = grid_map2.unsqueeze(1).repeat(1, weighted_second_feature.shape[1], 1, 1)
        second_pred = torch.mul(weighted_second_feature, repeated_grid)
        for i, layer in enumerate(self.second_map_conv2d):
            second_pred = layer(second_pred)
            if i == 4:
                upsampled_second_feat = self.upsample(second_pred)

        # First stage (small pedestrians)
        first_feature = torch.cat((multi_scale_features[features_names[0]], upsampled_second_feat), dim=1)

        # channel-wise attention module
        weighted_first_feature = self.attention_module[0](first_feature)

        # apply grid classifier
        grid_map1 = self.grid_classifiers[0](weighted_first_feature).view(-1,
                                                                          weighted_first_feature.shape[2],
                                                                          weighted_first_feature.shape[3])

        # spatial attention module
        repeated_grid = grid_map1.unsqueeze(1).repeat(1, weighted_first_feature.shape[1], 1, 1)
        first_pred = torch.mul(weighted_first_feature, repeated_grid)
        for i, layer in enumerate(self.first_map_conv2d):
            first_pred = layer(first_pred)

        num_samples = first_pred.size(0)
        self.grid_size1 = first_pred.size(2)
        self.grid_size2 = second_pred.size(2)
        self.grid_size3 = third_pred.size(2)

        first_yolo_prediction = (
            first_pred.view(num_samples, self.num_anchors, self.num_predictions, self.grid_size1, self.grid_size1)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        second_yolo_prediction = (
            second_pred.view(num_samples, self.num_anchors, self.num_predictions, self.grid_size2, self.grid_size2)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        third_yolo_prediction = (
            third_pred.view(num_samples, self.num_anchors, self.num_predictions, self.grid_size3, self.grid_size3)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # Get first yolo outputs
        first_x = first_yolo_prediction[..., 0]  # Center x
        first_y = first_yolo_prediction[..., 1]  # Center y
        first_w = first_yolo_prediction[..., 2]  # Width
        first_h = first_yolo_prediction[..., 3]  # Height
        first_pred_cls = first_yolo_prediction[..., 4:]  # Cls pred.
        first_pred_boxes = torch.stack((first_x, first_y, first_w, first_h), dim=-1)

        # Get second yolo outputs
        second_x = second_yolo_prediction[..., 0]  # Center x
        second_y = second_yolo_prediction[..., 1]  # Center y
        second_w = second_yolo_prediction[..., 2]  # Width
        second_h = second_yolo_prediction[..., 3]  # Height
        second_pred_cls = second_yolo_prediction[..., 4:]  # Cls pred.
        second_pred_boxes = torch.stack((second_x, second_y, second_w, second_h), dim=-1)

        # Get third yolo outputs
        third_x = third_yolo_prediction[..., 0]  # Center x
        third_y = third_yolo_prediction[..., 1]  # Center y
        third_w = third_yolo_prediction[..., 2]  # Width
        third_h = third_yolo_prediction[..., 3]  # Height
        third_pred_cls = third_yolo_prediction[..., 4:]  # Cls pred.
        third_pred_boxes = torch.stack((third_x, third_y, third_w, third_h), dim=-1)

        pred_dicts = {
            "first_pred_boxes": first_pred_boxes,
            "first_pred_cls": first_pred_cls,
            "second_pred_boxes": second_pred_boxes,
            "second_pred_cls": second_pred_cls,
            "third_pred_boxes": third_pred_boxes,
            "third_pred_cls": third_pred_cls,
            "grid_map1": grid_map1,
            "grid_map2": grid_map2,
            "grid_map3": grid_map3,
        }

        return pred_dicts

    def loss(self, example, pred_dicts, **kwargs):
        """
        Compute YOLO loss
        """
        # first yolo map ground truths
        tboxes1 = example['yolo_map1'][..., :-1]
        tcls1 = example['yolo_map1'][..., -1].view(-1).long()

        # second yolo map ground truths
        tboxes2 = example['yolo_map2'][..., :-1]
        tcls2 = example['yolo_map2'][..., -1].view(-1).long()

        # third yolo map ground truths
        tboxes3 = example['yolo_map3'][..., :-1]
        tcls3 = example['yolo_map3'][..., -1].view(-1).long()

        # grid classifiers ground truth
        tmap1 = example['classifier_map1']
        tmap2 = example['classifier_map2']
        tmap3 = example['classifier_map3']

        # first YOLO map predictions
        pred_boxes1 = pred_dicts["first_pred_boxes"]
        pred_cls1 = pred_dicts["first_pred_cls"].view(-1, 2)

        # second YOLO map predictions
        pred_boxes2 = pred_dicts["second_pred_boxes"]
        pred_cls2 = pred_dicts["second_pred_cls"].view(-1, 2)

        # third YOLO map predictions
        pred_boxes3 = pred_dicts["third_pred_boxes"]
        pred_cls3 = pred_dicts["third_pred_cls"].view(-1, 2)

        # grid classifiers predictions
        pmap1 = pred_dicts["grid_map1"]
        pmap2 = pred_dicts["grid_map2"]
        pmap3 = pred_dicts["grid_map3"]

        "Visualize grid maps"
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

        batch_size = pred_boxes1.size(0)

        # positive and negative samples masks
        obj_mask1 = example["obj_mask1"]
        pos_mask1 = obj_mask1.view(-1)
        neg_mask1 = example["noobj_mask1"].view(-1).type(torch.float32)

        obj_mask2 = example["obj_mask2"]
        pos_mask2 = obj_mask2.view(-1)
        neg_mask2 = example["noobj_mask2"].view(-1).type(torch.float32)

        obj_mask3 = example["obj_mask3"]
        pos_mask3 = obj_mask3.view(-1)
        neg_mask3 = example["noobj_mask3"].view(-1).type(torch.float32)

        # Hard negative mining
        # TODO instead of hard negatives, choose all negative samples and give more weight to pedestrians
        hard_neg_mask1, max_value1 = hard_negative_mining(neg_mask1, pos_mask1, pred_cls1, batch_size, neg_ratio=3.)
        hard_neg_mask2, max_value2 = hard_negative_mining(neg_mask2, pos_mask2, pred_cls2, batch_size, neg_ratio=3.)
        hard_neg_mask3, max_value3 = hard_negative_mining(neg_mask3, pos_mask3, pred_cls3, batch_size, neg_ratio=3.)

        # Regression loss for dimension and offset
        loc_loss1 = self.bbox_loss(pred_boxes1, tboxes1, obj_mask1)
        loc_loss2 = self.bbox_loss(pred_boxes2, tboxes2, obj_mask2)
        loc_loss3 = self.bbox_loss(pred_boxes3, tboxes3, obj_mask3)
        sum_loc_loss = loc_loss1 + loc_loss2 + loc_loss3
        loc_loss = 0.5 * (torch.exp(-self.sigma_loc) * sum_loc_loss + self.sigma_loc)

        # Grid classifiers loss
        loss_grid1 = self.grid_loss(pmap1, tmap1)
        loss_grid2 = self.grid_loss(pmap2, tmap2)
        loss_grid3 = self.grid_loss(pmap3, tmap3)
        sum_grid_loss = loss_grid1 + loss_grid2 + loss_grid3
        grid_loss = 0.5 * (torch.exp(-self.sigma_grid) * sum_grid_loss + self.sigma_grid)

        # classification loss
        cls_loss1 = self.class_loss(pred_cls1, tcls1, pos_mask1, hard_neg_mask1)
        cls_loss2 = self.class_loss(pred_cls2, tcls2, pos_mask2, hard_neg_mask2)
        cls_loss3 = self.class_loss(pred_cls3, tcls3, pos_mask3, hard_neg_mask3)
        sum_cls_loss = cls_loss1 + cls_loss2 + cls_loss3
        cls_loss = 1.0 * (torch.exp(-self.sigma_class) * sum_cls_loss + self.sigma_class)

        loss = cls_loss + loc_loss + grid_loss
        pos_num = pos_mask1.sum()

        ret = {}
        ret.update({'loss': loss, 'cls_loss': sum_cls_loss, 'loc_loss': sum_loc_loss, 'grid_loss': sum_grid_loss,
                    'sigma_class': self.sigma_class, 'sigma_loc': self.sigma_loc, 'sigma_grid': self.sigma_grid,
                    'num_pedestrians': pos_num, 'max_hard_value': max_value1})

        return ret

    @torch.no_grad()
    def predict(self, example, pred_dicts, test_cfg, **kwargs):
        """
        decode, nms, then return the detection result.
        """
        # get loss info
        prediction1 = pred_dicts["first_pred_boxes"]
        x1 = prediction1[..., 0]
        y1 = prediction1[..., 1]
        w1 = prediction1[..., 2]
        h1 = prediction1[..., 3]
        pred_cls1 = pred_dicts["first_pred_cls"]

        batch_size = prediction1.size(0)

        prediction2 = pred_dicts["second_pred_boxes"]
        x2 = prediction2[..., 0]
        y2 = prediction2[..., 1]
        w2 = prediction2[..., 2]
        h2 = prediction2[..., 3]
        pred_cls2 = pred_dicts["second_pred_cls"]

        prediction3 = pred_dicts["third_pred_boxes"]
        x3 = prediction3[..., 0]
        y3 = prediction3[..., 1]
        w3 = prediction3[..., 2]
        h3 = prediction3[..., 3]
        pred_cls3 = pred_dicts["third_pred_cls"]

        pmap1 = pred_dicts["grid_map1"]
        pmap2 = pred_dicts["grid_map2"]
        pmap3 = pred_dicts["grid_map3"]

        # pred_map1_vis = pmap1[0].cpu().detach().numpy()
        # pred_map2_vis = pmap2[0].cpu().detach().numpy()
        # pred_map3_vis = pmap3[0].cpu().detach().numpy()
        # pred_map1_vis = cv2.resize(pred_map1_vis, (self.img_dim, self.img_dim))
        # pred_map2_vis = cv2.resize(pred_map2_vis, (self.img_dim, self.img_dim))
        # pred_map3_vis = cv2.resize(pred_map3_vis, (self.img_dim, self.img_dim))
        # cv2.imshow("predicted_grid_map1", pred_map1_vis)
        # cv2.imshow("predicted_grid_map2", pred_map2_vis)
        # cv2.imshow("predicted_grid_map3", pred_map3_vis)
        # cv2.waitKey(0)

        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]

        # Add offset and scale with anchors
        grid_x1, grid_y1, anchor_w1, anchor_h1, stride1 = self._compute_grid_offsets(grid_size=self.grid_size1,
                                                                                     valid_anchors=[0, 1, 2],
                                                                                     cuda=prediction1.is_cuda)

        FloatTensor = torch.cuda.FloatTensor if prediction1.is_cuda else torch.FloatTensor
        pred_boxes1 = FloatTensor(prediction1[..., :4].shape)
        pred_boxes1[..., 0] = x1 * anchor_w1 + grid_x1
        pred_boxes1[..., 1] = y1 * anchor_h1 + grid_y1
        pred_boxes1[..., 2] = torch.exp(w1) * anchor_w1
        pred_boxes1[..., 3] = torch.exp(h1) * anchor_h1
        pred_cls1 = torch.nn.Softmax(dim=-1)(pred_cls1)

        grid_x2, grid_y2, anchor_w2, anchor_h2, stride2 = self._compute_grid_offsets(grid_size=self.grid_size2,
                                                                                     valid_anchors=[3, 4, 5],
                                                                                     cuda=prediction2.is_cuda)
        FloatTensor = torch.cuda.FloatTensor if prediction2.is_cuda else torch.FloatTensor
        pred_boxes2 = FloatTensor(prediction2[..., :4].shape)
        pred_boxes2[..., 0] = x2 * anchor_w2 + grid_x2
        pred_boxes2[..., 1] = y2 * anchor_h2 + grid_y2
        pred_boxes2[..., 2] = torch.exp(w2) * anchor_w2
        pred_boxes2[..., 3] = torch.exp(h2) * anchor_h2
        pred_cls2 = torch.nn.Softmax(dim=-1)(pred_cls2)

        grid_x3, grid_y3, anchor_w3, anchor_h3, stride3 = self._compute_grid_offsets(grid_size=self.grid_size3,
                                                                                     valid_anchors=[6, 7, 8],
                                                                                     cuda=prediction3.is_cuda)
        FloatTensor = torch.cuda.FloatTensor if prediction3.is_cuda else torch.FloatTensor
        pred_boxes3 = FloatTensor(prediction3[..., :4].shape)
        pred_boxes3[..., 0] = x3 * anchor_w3 + grid_x3
        pred_boxes3[..., 1] = y3 * anchor_h3 + grid_y3
        pred_boxes3[..., 2] = torch.exp(w3) * anchor_w3
        pred_boxes3[..., 3] = torch.exp(h3) * anchor_h3
        pred_cls3 = torch.nn.Softmax(dim=-1)(pred_cls3)

        grid_map1 = self.upsample_grid(pmap1.unsqueeze(1))
        grid_map2 = self.upsample_grid(pmap2.unsqueeze(1))
        grid_map3 = self.upsample_grid(pmap3.unsqueeze(1))
        grid_maps = torch.cat((grid_map1, grid_map2, grid_map3), dim=1)
        grid_maps = torch.mean(grid_maps, dim=1)

        output1 = torch.cat(
            (
                pred_boxes1.view(batch_size, -1, 4) * stride1,
                pred_cls1.view(batch_size, -1, 2),
            ),
            -1,
        )
        output2 = torch.cat(
            (
                pred_boxes2.view(batch_size, -1, 4) * stride2,
                pred_cls2.view(batch_size, -1, 2),
            ),
            -1,
        )
        output3 = torch.cat(
            (
                pred_boxes3.view(batch_size, -1, 4) * stride3,
                pred_cls3.view(batch_size, -1, 2),
            ),
            -1,
        )

        outputs = torch.cat((output1, output2, output3), dim=1)
        rets = self.post_processing(outputs, test_cfg)

        return rets

    @torch.no_grad()
    def post_processing(self, prediction, test_cfg):
        """
            Removes detections with lower object confidence score than 'conf_thres' and performs
            Non-Maximum Suppression to further filter detections.
            Returns detections with shape: (x1, y1, x2, y2, object_conf)
        """
        conf_thres = test_cfg.conf_thres
        nms_thres = test_cfg.nms_thres
        max_det = test_cfg.max_det
        max_nms = test_cfg.max_nms

        output = [torch.zeros((0, 5), device="cpu")] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[x[..., -1] > conf_thres]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = box_np_ops.xywh2xyxy(x[:, :4])
            conf = x[:, -1]
            x = torch.cat((box, conf[:, None]), 1)

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # sort by confidence
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # Batched NMS
            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, nms_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i].detach().cpu()

        return output
