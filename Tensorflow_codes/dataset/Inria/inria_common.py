import numpy as np
import pickle

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import cv2
import glob
import random
from scipy.io import loadmat
from collections import defaultdict

general_to_detection = {
    "person": "pedestrian",
    "people": "pedestrian",
    "person-fa": "ignore",
    "person?": "ignore"
}

def _fill_trainval_infos(root_path):
    train_cal_infos = []
    test_cal_infos = []

    train_pic_dir = os.path.join(root_path, "PICTURES_LABELS_TRAIN/PICTURES/" + "*.jpg")
    train_label_dir = os.path.join(root_path, "PICTURES_LABELS_TRAIN/ANOTATION")
    for image_path in sorted(glob.glob(train_pic_dir)):

        filename = os.path.basename(image_path)
        basefile = filename.split(".")[0]
        label_name = os.path.join(train_label_dir, (basefile + '.xml'))

        DOMTree = xml.dom.minidom.parse(label_name)
        collection = DOMTree.documentElement
        objs = collection.getElementsByTagName("object")
        gt_boxes = []
        gt_names = []
        for obj in objs:
            obj_type = obj.getElementsByTagName('name')[0].childNodes[0].data
            name = general_to_detection[obj_type]
            if name == "pedestrian":
                bbox = obj.getElementsByTagName('bndbox')[0]
                ymin = float(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmin = float(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymax = float(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
                xmax = float(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
                gt_boxes.append((int(ymin), int(xmin), int(ymax), int(xmax)))
                gt_names.append(name)

        if len(gt_boxes) == 0:
            continue

        infos = {'image_path': image_path, 'gt_boxes': np.stack(gt_boxes, axis=0),
                 'gt_names': gt_names}

        # img = cv2.imread(image_path)
        # for cx, cy, w, h in gt_boxes:
        #     x1 = int(cx - w // 2)
        #     y1 = int(cy - h // 2)
        #     x2 = int(cx + w // 2)
        #     y2 = int(cy + h // 2)
        #     cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)
        #
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

        train_cal_infos.append(infos)

    test_pic_dir = os.path.join(root_path, "PICTURES_LABELS_TEST/PICTURES/" + "*.jpg")
    test_label_dir = os.path.join(root_path, "PICTURES_LABELS_TEST/ANOTATION")
    for image_path in sorted(glob.glob(test_pic_dir)):

        filename = os.path.basename(image_path)
        basefile = filename.split(".")[0]
        label_name = os.path.join(test_label_dir, (basefile + '.xml'))

        DOMTree = xml.dom.minidom.parse(label_name)
        collection = DOMTree.documentElement
        objs = collection.getElementsByTagName("object")
        gt_boxes = []
        gt_names = []
        for obj in objs:
            obj_type = obj.getElementsByTagName('name')[0].childNodes[0].data
            name = general_to_detection[obj_type]
            if name == "pedestrian":
                bbox = obj.getElementsByTagName('bndbox')[0]
                ymin = float(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmin = float(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymax = float(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
                xmax = float(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
                gt_boxes.append((int(ymin), int(xmin), int(ymax), int(xmax)))
                gt_names.append(name)

        if len(gt_boxes) == 0:
            continue

        infos = {'image_path': image_path, 'gt_boxes': np.stack(gt_boxes, axis=0),
                 'gt_names': gt_names}

        test_cal_infos.append(infos)

    return train_cal_infos, test_cal_infos


def create_inria_infos(root_path):

    train_cal_infos, test_cal_infos = _fill_trainval_infos(root_path)

    print(f"train sample: {len(train_cal_infos)}, test sample: {len(test_cal_infos)}")

    with open(root_path + "/" + "infos_train.pkl", "wb") as f:
        pickle.dump(train_cal_infos, f)
    with open(root_path + "/" + "infos_test.pkl", "wb") as f:
        pickle.dump(test_cal_infos, f)
