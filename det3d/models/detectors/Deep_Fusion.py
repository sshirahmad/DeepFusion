from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy
import cv2
import torch


@DETECTORS.register_module
class DeepFusion(SingleStageDetector):
    def __init__(
        self,
        bbox_head,
        backbone_image=None,
        backbone_radar=None,
        reader=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(DeepFusion, self).__init__(
            backbone_image=backbone_image, backbone_radar=backbone_radar, reader=reader, bbox_head=bbox_head,
            train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained
        )

    def extract_feat(self, data, **kwargs):
        contain_pcd = kwargs.get("contain_pcd", None)
        # TODO change this for sparse
        if contain_pcd:
            radar_features = self.reader(data["features"], data["num_voxels"], data["coors"])
            x_radar = self.backbone_radar(radar_features, data["coors"], data["batch_size"], data["input_shape"])
            x_cat = torch.cat((data["image"], x_radar), dim=1)
            x = self.backbone_image(x_cat)

            "Visualize dense radar Pseudo images"
            # vis_radar = x_radar.cpu().detach().numpy()
            # vis_image = data["image"].cpu().detach().numpy()
            # cv2.imwrite("radar_pseudo_image.png", vis_radar[0].reshape(416, 416, 3) * 255)
            # cv2.imwrite("image2.png", vis_image[0].reshape(416, 416, 3) * 255)
            # cv2.waitKey(0)

        else:
          x = self.backbone_image(data["image"])

        return x

    def forward(self, example, return_loss=True, **kwargs):
        contain_pcd = kwargs.get("contain_pcd", None)
        if contain_pcd:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]
            image = example["image"]
            batch_size = len(num_voxels)

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                image=image,
                batch_size=batch_size,
                input_shape=example["shape"][0],
            )
        else:
            image = example["image"]

            data = dict(
                image=image,
                batch_size=image.shape[0],
                input_shape=image.shape[1:],
            )

        multi_scale_features = self.extract_feat(data, **kwargs)
        pred_dicts = self.bbox_head(multi_scale_features)

        if return_loss:
            return self.bbox_head.loss(example, pred_dicts, **kwargs)
        else:
            return self.bbox_head.predict(example, pred_dicts, self.test_cfg)
