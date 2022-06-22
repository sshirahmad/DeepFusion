import collections
from collections import defaultdict
import numpy as np
import torch


def collate(batch_list):
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if type(example) is list:
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))

        elif key == "metadata":
            ret[key] = elems

        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))

        elif key == "gt_boxes":
            # Add sample index to targets
            for i, boxes in enumerate(elems):
                boxes[:, 0] = i
            ret[key] = torch.cat(elems, 0)

        elif key in ["yolo_map1", "yolo_map2", "yolo_map3",
                     "classifier_map1", "classifier_map2", "classifier_map3",
                     "obj_mask1", "obj_mask2", "obj_mask3", "noobj_mask1", "noobj_mask2",
                     "noobj_mask3"]:
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        elif key in ["image"]:
            ret[key] = torch.stack(elems)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret

