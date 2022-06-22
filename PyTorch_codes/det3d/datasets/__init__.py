from .builder import build_dataset

# from .cityscapes import CityscapesDataset
from .nuscenes import NuScenesDataset
from .caltech import CaltechDataset
from .Inria import InriaDataset
from .transforms import *
from .augmentations import *

# from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset

# from .extra_aug import ExtraAugmentation
from .loader import build_dataloader
from .registry import DATASETS

# from .voc import VOCDataset
# from .wider_face import WIDERFaceDataset
# from .xml_style import XMLDataset
#
__all__ = [
    "resize",
    "build_dataloader",
    "ConcatDataset",
    "RepeatDataset",
    "DATASETS",
    "build_dataset",
]
