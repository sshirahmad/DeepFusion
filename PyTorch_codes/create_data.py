import copy
from pathlib import Path
import pickle
import numpy as np
import argparse

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.caltech import caltech_common as ca_ds
from det3d.datasets.Inria import inria_common as in_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.utils.create_anchors import generate_anchors, generate_anchors_caltech
from det3d.torchie import Config


def nuscenes_data_prep(root_path, version, nsweeps=10, img_size=416, filter_zero=True):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    if version != "v1.0-test":
        info_path = Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero)
        # create anchors
        with open(info_path, 'rb') as f:
            train_dataset = pickle.load(f)

        anchors_path = Path(root_path) / "anchors.txt"
        anchors = generate_anchors(train_dataset, num_anchors=9, img_size=img_size, class_name="pedestrian")
        np.savetxt(anchors_path, anchors)


def caltech_data_prep(root_path, img_size):
    ca_ds.create_caltech_infos(root_path)
    info_path = Path(root_path) / "infos_train.pkl"

    # create anchors
    with open(info_path, 'rb') as f:
        train_dataset = pickle.load(f)

    anchors_path = Path(root_path) / "anchors.txt"
    anchors = generate_anchors_caltech(train_dataset, num_anchors=9, img_size=img_size, class_name="pedestrian")
    np.savetxt(anchors_path, anchors)


def inria_data_prep(root_path, img_size):
    in_ds.create_inria_infos(root_path)
    info_path = Path(root_path) / "infos_train.pkl"
    info_path_test = Path(root_path) / "infos_test.pkl"

    # create anchors
    with open(info_path, 'rb') as f:
        train_dataset = pickle.load(f)

    # create anchors
    with open(info_path_test, 'rb') as f:
        test_dataset = pickle.load(f)

    dataset = train_dataset + test_dataset
    anchors_path = Path(root_path) / "anchors.txt"
    anchors = generate_anchors_caltech(dataset, num_anchors=9, img_size=img_size, class_name="pedestrian")
    np.savetxt(anchors_path, anchors)


parser = argparse.ArgumentParser(description="Train a detector")
parser.add_argument("--config", default="./configs/mobilenetv3_yolov3_inria.py", help="train config file path")
parser.add_argument("--root", help="path to dataset directory")
parser.add_argument("--version", help="version of the dataset (NuScenes)")



if __name__ == "__main__":
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    nuscenes_data_prep(root_path=args.root, version=args.version, img_size=cfg.img_dim, nsweeps=10, filter_zero=True)

    # caltech_data_prep(root_path, cfg.img_dim)

    # inria_data_prep(root_path, cfg.img_dim)

