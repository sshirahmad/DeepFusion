import numpy as np
from PIL import Image
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from imgaug import augmenters as iaa
from det3d.core.input.voxel_generator import VoxelGenerator
from ..registry import PIPELINES
import cv2


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.img_size = cfg.img_size
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.contain_pcd = cfg.contain_pcd

        self.mode = cfg.mode
        self.pad_square = iaa.Sequential([iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()])
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_noise = cfg.get('global_translate_noise', 0)
            self.image_aug = iaa.Sequential(cfg.image_aug_seq)
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["NuScenesDataset"]:
            points = res["radar"]["points"]
            img = np.array(Image.open(res["camera"]["image"]).convert('RGB'), dtype=np.uint8)
        elif res["type"] in ["CaltechDataset", "InriaDataset"]:
            img = np.array(Image.open(res["camera"]["image"]).convert('RGB'), dtype=np.uint8)
        else:
            raise NotImplementedError

        boxes = res["camera"]["annotations"]["boxes"]
        if self.contain_pcd:
            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(points, boxes)
                mask = point_counts >= self.min_points_in_gt
                boxes = boxes[mask]

        if self.mode == "train" and not self.no_augmentation:
            # Augment dataset
            if self.contain_pcd:
                points_aug, enable = prep.random_flip(points)
                points_aug, noise_rotation = prep.global_rotation(points_aug, rotation=self.global_rotation_noise)
                points_aug, noise_scale = prep.global_scaling(points_aug, self.global_scaling_noise)
                points_aug, noise_translate = prep.global_translate(points_aug, noise_translate_std=self.global_translate_noise)

                if enable:
                    self.radar_aug_seq = [
                                        iaa.Fliplr(1.0),
                                        iaa.Affine(
                                            scale=noise_scale,
                                            translate_px={"x": int(noise_translate[0]), "y": int(noise_translate[1])},
                                            rotate=noise_rotation,
                                        ),
                                    ]
                else:
                    self.radar_aug_seq = [
                        iaa.Fliplr(),
                        iaa.Affine(
                            scale=noise_scale,
                            translate_px={"x": int(noise_translate[0]), "y": int(noise_translate[1])},
                            rotate=noise_rotation,
                        ),
                    ]

                radar_augmentations = iaa.Sequential(self.radar_aug_seq, random_order=True)
                boxes_aug, img_aug = prep.ImgAug(img, boxes, radar_augmentations)

                image_augmentations = iaa.Sequential(self.image_aug_seq, random_order=True)
                boxes_aug, img_aug[:, :, 0:3] = prep.ImgAug(img_aug[:, :, 0:3], boxes_aug, image_augmentations)
                if len(boxes) != 0:
                    boxes = boxes_aug
                    img = img_aug
                    points = points_aug
            else:
                boxes_aug, img_aug = prep.ImgAug(img, boxes, self.image_aug)
                if len(boxes_aug) != 0:
                    boxes = boxes_aug
                    img = img_aug
                    # for box in boxes:
                    #     cv2.rectangle(img_aug, (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)),
                    #                   (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)), (255, 0, 0), 2)
                    # cv2.imshow('image', img_aug)
                    # cv2.waitKey(0)

        # make square images
        boxes, img = prep.ImgAug(img, boxes, self.pad_square)

        # normalize images and bounding boxes
        height, width, _ = img.shape
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height

        # resize the images
        if res["camera"]["image_resize"] is not None:
            img = cv2.resize(img, dsize=(res["camera"]["image_resize"], res["camera"]["image_resize"]))
        else:
            img = cv2.resize(img, dsize=(self.img_size, self.img_size))

        # normalize the images
        height, width, channels = img.shape
        img = img.reshape(channels, height, width)
        res["camera"]["image"] = img / 255.0

        if self.contain_pcd:
            if self.shuffle_points:
                np.random.shuffle(points)
            res["radar"]["points"] = points

        res["camera"]["annotations"] = boxes

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num,
                                                                                  int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["camera"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            # mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            # _dict_select(gt_dict, mask)

            res["camera"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["radar"]["points"], max_voxels=max_voxels
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["radar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["radar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["radar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["radar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["radar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["radar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["radar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        return res, info


@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):

        example = {}

        boxes = res["camera"]["annotations"]
        _, _, grid_size = res["camera"]["image"].shape

        nG = list([grid_size // 8, grid_size // 16, grid_size // 32])
        corner_boxes = prep.xywh2xyxy_np(boxes)

        # Create grid classifiers targets
        grid_maps = []
        for k, size in enumerate(nG):
            grid_map = np.zeros(shape=(size, size), dtype=np.float32)
            x = np.arange(size, dtype=np.float32)
            y = x.T
            for i in range(size):
                for j in range(size):
                    box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                    iou = box_np_ops.overlap_jit(corner_boxes * size, box)
                    grid_map[j, i] = iou.max(0)

            grid_maps.append(grid_map)

        example.update({
            'gt_boxes': boxes,
            'classifier_map1': grid_maps[0],
            'classifier_map2': grid_maps[1],
            'classifier_map3': grid_maps[2],
        })

        res["camera"]["targets"] = example

        return res, info