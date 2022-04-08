import torch.nn as nn
import torch
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone_image=None,
        backbone_radar=None,
        reader=None,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        if backbone_image is not None:
            self.backbone_image = builder.build_backbone(backbone_image)
        if backbone_radar is not None:
            self.backbone_radar = builder.build_backbone(backbone_radar)
        if reader is not None:
            self.reader = builder.build_reader(reader)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self.backbone_image, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data, **kwargs):
        contain_pcd = kwargs.get("contain_pcd", None)
        if contain_pcd:
            input_features = self.reader(
                data["features"], data["num_voxels"], data["coors"]
            )
            x_radar = self.backbone_radar(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )
            image_radar = torch.cat(x_radar, data["image"])
            x = self.backbone_image(image_radar)
        else:
            x = self.backbone_image(
                data["image"]
            )
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self