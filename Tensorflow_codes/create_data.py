import copy
from pathlib import Path
import pickle
import numpy as np
from utils.config import Config

from dataset.Inria import inria_common as in_ds

def inria_data_prep(root_path, img_size):
    in_ds.create_inria_infos(root_path)
    info_path = Path(root_path) / "infos_train.pkl"

    # # create anchors
    # with open(info_path, 'rb') as f:
    #     train_dataset = pickle.load(f)
    #
    # anchors_path = Path(root_path) / "anchors.txt"
    # anchors = generate_anchors_caltech(train_dataset, num_anchors=2, img_size=img_size, class_name="pedestrian")
    # np.savetxt(anchors_path, anchors)


if __name__ == "__main__":
    config_path = "./config.py"
    cfg = Config.fromfile(config_path)

    # DATA_ROOT = "C:/MSc_thesis/Dataset/Nuscenes/nuScenes"
    # version = "v1.0-mini"
    # nuscenes_data_prep(root_path=DATA_ROOT, version=version, img_size=cfg.img_dim, nsweeps=10, filter_zero=True)

    # root_path = "C:/MSc_thesis/Dataset/Caltech"
    # caltech_data_prep(root_path, cfg.img_dim)

    root_path = "C:/MSc_thesis/Dataset/inria_person"
    inria_data_prep(root_path, cfg.img_size)

