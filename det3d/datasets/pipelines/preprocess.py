import numpy as np
from PIL import Image
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.core.bbox import box_torch_ops
from det3d.builder import build_dbsampler
from imgaug import augmenters as iaa
import imgaug as ia
import torch
import math
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
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
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_noise = cfg.get('global_translate_noise', 0)
            self.image_aug_seq = cfg.image_aug_seq
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None

            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["NuScenesDataset"]:
            points = res["radar"]["points"]
            radar_img = res["radar"]["image"]
            img = np.array(Image.open(res["camera"]["image"]).convert('RGB'), dtype=np.uint8)
            # img = np.concatenate((cam_img, radar_img), axis=2) #TODO change this for sparse
        elif res["type"] in ["CaltechDataset", "InriaDataset"]:
            img = np.array(Image.open(res["camera"]["image"]).convert('RGB'), dtype=np.uint8)
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["camera"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            if gt_dict["gt_boxes"].shape[1] > 4:
                # convert 3D boxes to 2D boxes
                intrinsic = info["cam_intrinsic"]
                gt_dict["gt_boxes"] = box_np_ops.convert_box3d_box2d(gt_dict["gt_boxes"], intrinsic).reshape(-1, 4)

            if self.contain_pcd:
                if self.min_points_in_gt > 0:
                    point_counts = box_np_ops.points_count_rbbox(
                        points, gt_dict["gt_boxes"]
                    )
                    mask = point_counts >= self.min_points_in_gt
                    _dict_select(gt_dict, mask)

            # replace class names with numbers
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

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
                bboxes, img_aug = prep.ImgAug(img, gt_dict["gt_boxes"],
                                          gt_dict["gt_names"], radar_augmentations)

                image_augmentations = iaa.Sequential(self.image_aug_seq, random_order=True)
                bboxes, img_aug[:, :, 0:3] = prep.ImgAug(img_aug[:, :, 0:3], bboxes,
                                                gt_dict["gt_names"], image_augmentations)
                if len(bboxes) != 0:
                    # sometimes aug func will crop no person
                    # logger.warning("No person img, abandoned...")
                    gt_dict["gt_boxes"] = bboxes
                    img = img_aug
                    points = points_aug
            else:
                while True:
                    augmentations = iaa.SomeOf(3, self.image_aug_seq, random_order=True)
                    bboxes, img_aug = prep.ImgAug(img, gt_dict["gt_boxes"], gt_dict["gt_names"], augmentations)

                    if len(bboxes) == 0:
                        continue
                    else:
                        x_ = (bboxes[:, 2] - bboxes[:, 0]) / img.shape[1]
                        y_ = (bboxes[:, 3] - bboxes[:, 1]) / img.shape[0]
                        mask = (x_ > 0.1) & (y_ > 0.1)
                        if mask.any():
                            gt_dict["gt_boxes"] = bboxes[mask]
                            img = img_aug
                            # for box in bboxes:
                            #     cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                            # cv2.imshow('image', img_aug)
                            # cv2.waitKey(0)
                            break
                        else:
                            continue

        elif self.no_augmentation:
            if gt_dict["gt_boxes"].shape[1] > 4:
                # convert 3D boxes to 2D boxes
                intrinsic = info["cam_intrinsic"]
                gt_dict["gt_boxes"] = box_np_ops.convert_box3d_box2d(gt_dict["gt_boxes"], intrinsic).reshape(-1, 4)

            if self.contain_pcd:
                if self.min_points_in_gt > 0:
                    point_counts = box_np_ops.points_count_rbbox(
                        points, gt_dict["gt_boxes"]
                    )
                    mask = point_counts >= self.min_points_in_gt
                    _dict_select(gt_dict, mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        # normalize images and bounding boxes
        if self.mode == "train":
            height, width, _ = img.shape
            gt_dict["gt_boxes"][:, 0] = gt_dict["gt_boxes"][:, 0] / width
            gt_dict["gt_boxes"][:, 1] = gt_dict["gt_boxes"][:, 1] / height
            gt_dict["gt_boxes"][:, 2] = gt_dict["gt_boxes"][:, 2] / width
            gt_dict["gt_boxes"][:, 3] = gt_dict["gt_boxes"][:, 3] / height
            # gt_dict["gt_boxes"] = np.clip(gt_dict["gt_boxes"], 0., 1.)

            if res["camera"]["image_resize"] is not None:
                img = cv2.resize(img, dsize=(res["camera"]["image_resize"], res["camera"]["image_resize"]))
            else:
                img = cv2.resize(img, dsize=(self.img_size, self.img_size))
            height, width, channels = img.shape
            img = img.reshape(channels, height, width)
            res["camera"]["image"] = (2.0 / 255.0) * img - 1.0
            # res["camera"]["image"] = img / 255.0
        else:
            img = cv2.resize(img, dsize=(self.img_size, self.img_size))
            height, width, channels = img.shape
            img = img.reshape(channels, height, width)
            res["camera"]["image"] = (2.0 / 255.0) * img - 1.0
            # res["camera"]["image"] = img / 255.0

        if self.contain_pcd:
            if self.shuffle_points:
                np.random.shuffle(points)
            res["radar"]["points"] = points

        if self.mode == "train":
            res["camera"]["annotations"] = gt_dict

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
class AssignLabelYOLO(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        self.tasks = assigner_cfg.target_assigner.tasks
        self.ignore_thres = assigner_cfg.ignore_thres
        self.grid_maps = assigner_cfg.grid_maps
        self.surrounding_size = assigner_cfg.surrounding_size
        self.top_k = assigner_cfg.top_k
        self.img_dim = assigner_cfg.img_dim
        self.anchors = assigner_cfg.anchors
        self.num_anchors = len(self.anchors)

    def __call__(self, res, info):

        example = {}
        if res["mode"] == "train":
            gt_dict = res["camera"]["annotations"]
            corner_bboxes = gt_dict["gt_boxes"]

            nA = self.num_anchors
            nG = self.grid_maps[-1]
            nG_maps = self.grid_maps

            # make target boxes relative to grid map
            target_boxes = corner_bboxes

            obj_mask = np.zeros(shape=(nA, nG, nG), dtype=np.bool_)
            noobj_mask = np.ones(shape=(nA, nG, nG), dtype=np.bool_)
            tx = np.zeros(shape=(nA, nG, nG), dtype=np.float32)
            ty = np.zeros(shape=(nA, nG, nG), dtype=np.float32)
            tw = np.zeros(shape=(nA, nG, nG), dtype=np.float32)
            th = np.zeros(shape=(nA, nG, nG), dtype=np.float32)
            tboxes = np.zeros(shape=(nA, nG, nG, 4), dtype=np.float32)

            grid_maps = []
            for size in nG_maps:
                x = np.arange(size, dtype=np.float32)
                y = x.T
                ious = np.zeros(shape=(size, size))
                targets = corner_bboxes * size
                for i in range(size):
                    for j in range(size):
                        box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                        iou = box_np_ops.overlap_jit(targets, box)
                        ious[j, i] = iou.max(0)
                grid_maps.append(ious)

            # create groundtruth of YOLO classifiers
            h_per_cell = 1 / nG
            w_per_cell = 1 / nG
            center_location_h_index = np.int32((target_boxes[:, 1] + target_boxes[:, 3]) / (2 * h_per_cell))
            center_location_w_index = np.int32((target_boxes[:, 0] + target_boxes[:, 2]) / (2 * w_per_cell))

            stride = 1 / self.img_dim
            scaled_anchors = self.anchors * stride

            # find the index of surrounding priori boxes
            priori_box_index = []
            total_ovr_info = []
            for iter in range(len(center_location_w_index)):
                if center_location_h_index[iter] - self.surrounding_size >= 0:
                    min_h_index = center_location_h_index[iter] - self.surrounding_size
                else:
                    min_h_index = 0
                if center_location_h_index[iter] + self.surrounding_size <= nG - 1:
                    max_h_index = center_location_h_index[iter] + self.surrounding_size
                else:
                    max_h_index = nG - 1
                if center_location_w_index[iter] - self.surrounding_size >= 0:
                    min_w_index = center_location_w_index[iter] - self.surrounding_size
                else:
                    min_w_index = 0
                if center_location_w_index[iter] + self.surrounding_size <= nG - 1:
                    max_w_index = center_location_w_index[iter] + self.surrounding_size
                else:
                    max_w_index = nG - 1

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

                # sort ovr_infos in decreasing IoU order
                inds = np.argsort(ovr_info[:, 3])[::-1]

                # find k best priori box matches for each bounding box
                num = 0
                for index in inds:
                    info = [ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]]
                    if info not in priori_box_index:  # to avoid same priorBox match multi groundTruth
                        if num >= self.top_k:  # this value means we choose the top-k ovr box to be the positive box
                            break
                        priori_box_index.append([ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]])
                        num += 1
            priori_box_index = np.array(priori_box_index, dtype=np.int32)
            priori_box_index = priori_box_index.reshape([-1, self.top_k, 3])
            total_ovr_info = np.concatenate(total_ovr_info)

            for ground_truth_index in range(len(priori_box_index)):
                target_bbox = target_boxes[ground_truth_index]
                for index_info in priori_box_index[ground_truth_index]:
                    y_c_anchor = (index_info[0] + 0.5) * h_per_cell
                    x_c_anchor = (index_info[1] + 0.5) * w_per_cell
                    h_anchor = scaled_anchors[index_info[2], 1]
                    w_anchor = scaled_anchors[index_info[2], 0]

                    y_c_gt = (target_bbox[3] + target_bbox[1]) / 2
                    x_c_gt = (target_bbox[0] + target_bbox[2]) / 2
                    h_gt = target_bbox[3] - target_bbox[1]
                    w_gt = target_bbox[2] - target_bbox[0]

                    # difference between anchor boxes and groundtruth
                    y_t = (y_c_gt - y_c_anchor) / h_anchor
                    x_t = (x_c_gt - x_c_anchor) / w_anchor
                    h_t = np.log(h_gt / h_anchor + 1e-16)
                    w_t = np.log(w_gt / w_anchor + 1e-16)

                    tx[index_info[2], index_info[0], index_info[1]] = x_t
                    ty[index_info[2], index_info[0], index_info[1]] = y_t
                    tw[index_info[2], index_info[0], index_info[1]] = w_t
                    th[index_info[2], index_info[0], index_info[1]] = h_t
                    tboxes[index_info[2], index_info[0], index_info[1]] = target_bbox

                    obj_mask[index_info[2], index_info[0], index_info[1]] = 1
                    noobj_mask[index_info[2], index_info[0], index_info[1]] = 0

                    tcls = obj_mask.astype(np.float)
                    gt_boxes_cls = np.stack((tx, ty, tw, th, tcls), axis=-1)

            "visualize preprocessed ground truths"
            # for i in range(len(nG_maps)):
            #     image = res['camera']['image'][0:3, :, :].reshape(self.img_dim, self.img_dim, -1)
            #     resized_grid = cv2.resize(grid_maps[i], (self.img_dim, self.img_dim))
            #     for box in corner_bboxes:
            #         box = box * self.img_dim
            #         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            #     cv2.imshow('image', image)
            #     cv2.imshow('grid', resized_grid)
            #     cv2.waitKey(0)

            example.update({'gt_boxes_cls': gt_boxes_cls,
                            'grid_map1': grid_maps[0],
                            'grid_map2': grid_maps[1],
                            'grid_map3': grid_maps[2],
                            'obj_mask': obj_mask,
                            'noobj_mask': noobj_mask})

        else:
            pass

        res["camera"]["targets"] = example

        return res, info


@PIPELINES.register_module
class AssignLabelYOLOv3(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        self.tasks = assigner_cfg.target_assigner.tasks
        self.obj_thres = assigner_cfg.obj_thres
        self.top_k = assigner_cfg.top_k
        self.surrounding_size = assigner_cfg.surrounding_size
        self.img_dim = assigner_cfg.img_dim
        self.anchors = assigner_cfg.anchors
        self.num_anchors = len(self.anchors)

    def __call__(self, res, info):

        example = {}
        if res["mode"] == "train":
            gt_dict = res["camera"]["annotations"]
            _, _, grid_size = res["camera"]["image"].shape
            corner_bboxes = gt_dict["gt_boxes"]

            nA = self.num_anchors
            nG = list([grid_size // 8, grid_size // 16, grid_size // 32])
            valid_anchors = list(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
            best_anchor_ind = [np.argmax(box_np_ops.bbox_wh_iou(gt_box, self.anchors / self.img_dim)) for gt_box in corner_bboxes]

            # Output tensors
            classifier_maps = []
            yolo_maps = []
            target_boxes_grid = []
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
                tboxes = np.zeros(shape=(nA // 3, size, size, 4), dtype=np.float32)
                grid_maps = np.zeros(shape=(size, size), dtype=np.float32)

                mask = np.array([best_anchor_ind[i] in valid_anchors[k] for i in range(len(best_anchor_ind))])
                if mask.any():
                    # create groundtruth of grid classifiers
                    x = np.arange(size, dtype=np.float32)
                    y = x.T
                    for i in range(size):
                        for j in range(size):
                            box = np.array([[x[i], y[j], x[i] + 1, y[j] + 1]])
                            iou = box_np_ops.overlap_jit(target_boxes[mask] * size, box)
                            grid_maps[j, i] = iou.max(0)

                # create groundtruth of YOLO classifiers
                h_per_cell = 1 / size
                w_per_cell = 1 / size
                center_location_h_index = np.int32((target_boxes[:, 1] + target_boxes[:, 3]) / (2 * h_per_cell))
                center_location_w_index = np.int32((target_boxes[:, 0] + target_boxes[:, 2]) / (2 * w_per_cell))

                stride = 1 / self.img_dim
                scaled_anchors = self.anchors[valid_anchors[k]] * stride

                # find the index of surrounding priori boxes
                total_ovr_info = []
                priori_box_index = []
                for iter in range(len(center_location_w_index)):
                    if center_location_h_index[iter] - self.surrounding_size >= 0:
                        min_h_index = center_location_h_index[iter] - self.surrounding_size
                    else:
                        min_h_index = 0
                    if center_location_h_index[iter] + self.surrounding_size <= size - 1:
                        max_h_index = center_location_h_index[iter] + self.surrounding_size
                    else:
                        max_h_index = size - 1
                    if center_location_w_index[iter] - self.surrounding_size >= 0:
                        min_w_index = center_location_w_index[iter] - self.surrounding_size
                    else:
                        min_w_index = 0
                    if center_location_w_index[iter] + self.surrounding_size <= size - 1:
                        max_w_index = center_location_w_index[iter] + self.surrounding_size
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
                            if num >= self.top_k:  # this value means we choose the top-k ovr box to be the positive box
                                break
                            priori_box_index.append([ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]])
                            num += 1

                priori_box_index = np.array(priori_box_index, dtype=np.int32)
                priori_box_index = priori_box_index.reshape([-1, self.top_k, 3])

                for ground_truth_index in range(len(total_ovr_info)):
                    target_bbox = target_boxes[ground_truth_index]
                    if best_anchor_ind[ground_truth_index] in valid_anchors[k]:
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
                            tboxes[anchor_ind, y_ind, x_ind] = target_bbox

                            obj_mask[anchor_ind, y_ind, x_ind] = 1
                            noobj_mask[anchor_ind, y_ind, x_ind] = 0

                    for index_info in total_ovr_info[ground_truth_index]:
                        if index_info[-1] >= self.obj_thres:
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
                            tboxes[anchor_ind, y_ind, x_ind] = target_bbox

                            obj_mask[anchor_ind, y_ind, x_ind] = 1
                            noobj_mask[anchor_ind, y_ind, x_ind] = 0

                tcls = obj_mask.astype(np.float)
                gt_boxes_cls = np.stack((tx, ty, tw, th, tcls), axis=-1)
                classifier_maps.append(grid_maps)
                yolo_maps.append(gt_boxes_cls)
                obj_masks.append(obj_mask)
                noobj_masks.append(noobj_mask)
                target_boxes_grid.append(tboxes)

            "visualize preprocessed ground truths"
            # for i in range(len(nG)):
            #     image = res['camera']['image'][0:3, :, :].reshape(self.img_dim,  self.img_dim, -1)
            #     resized_grid = cv2.resize(classifier_maps[i], (self.img_dim,  self.img_dim))
            #     for box in corner_bboxes:
            #         box = box * self.img_dim
            #         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            #     cv2.imshow('image', image)
            #     cv2.imshow('grid', resized_grid)
            #     cv2.waitKey(0)

            example.update({
                'yolo_map1': yolo_maps[0],
                'yolo_map2': yolo_maps[1],
                'yolo_map3': yolo_maps[2],
                'target_boxes_grid1': target_boxes_grid[0],
                'target_boxes_grid2': target_boxes_grid[1],
                'target_boxes_grid3': target_boxes_grid[2],
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

        else:
            pass

        res["camera"]["targets"] = example

        return res, info