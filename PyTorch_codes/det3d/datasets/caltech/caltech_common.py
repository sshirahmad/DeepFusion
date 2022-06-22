import numpy as np
import pickle

from pathlib import Path
from functools import reduce
from tqdm import tqdm
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


def _save_img(out_dir, set_id, fn, i, frame):
    im_path = '{}/{}_{}_{}.png'.format(out_dir, os.path.basename(set_id), os.path.basename(fn).split('.')[0], i)
    cv2.imwrite(im_path, frame)


def _convert_seq_to_images(root_path):
    input_path = root_path + "/" + "Sequences"
    output_path = root_path + "/" + "Images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # read images
    for dname in sorted(glob.glob(os.path.join(input_path, 'set*'))):

        set_id = os.path.basename(dname)

        for fn in tqdm(sorted(glob.glob('{}/*.seq'.format(dname)))):
            cap = cv2.VideoCapture(fn)
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _save_img(output_path, set_id, fn, frame_id, frame)
                frame_id += 1


def _fill_trainval_infos(root_path, train_sets):
    train_cal_infos = []
    test_cal_infos = []

    input_path = root_path + "/" + "Sequences"

    # read annotations
    for dname in sorted(glob.glob(os.path.join(input_path, 'set*'))):

        set_id = os.path.basename(dname)

        for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
            vbb = loadmat(anno_fn)
            obj_lists = vbb['A'][0][0][1][0]  # list of objects in whole video
            obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
            video_id = os.path.splitext(os.path.basename(anno_fn))[0]

            # traverse frames
            for frame_id, obj in enumerate(obj_lists):  # list of objects in each frame of video
                if len(obj) > 0:

                    gt_boxes = []
                    gt_names = []
                    for pedestrian_id, pedestrian_pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                        pedestrian_id = int(pedestrian_id[0][0]) - 1
                        occlusion = int(occl[0][0])
                        pedestrian_pos = pedestrian_pos[0].tolist()
                        name = obj_lbl[pedestrian_id]
                        mapped_name = general_to_detection[name]

                        # reasonable setting
                        if mapped_name == "pedestrian" and occlusion == 0 and 50 < pedestrian_pos[3]:
                            (box_x_left, box_y_top, box_w, box_h) = pedestrian_pos
                            cx = (box_x_left + box_w / 2.0)
                            cy = (box_y_top + box_h / 2.0)
                            w = box_w
                            h = box_h
                            gt_boxes.append((cx, cy, w, h))
                            gt_names.append(general_to_detection[name])

                    if len(gt_boxes) == 0:
                        continue

                    infos = {
                        "image_path": []
                    }

                    image_path = root_path + "/" + "Images" + "/" + set_id + '_' + video_id + '_' + str(frame_id) + '.png'

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

                    infos['image_path'] = image_path
                    infos['gt_boxes'] = np.stack(gt_boxes, axis=0)
                    infos['gt_names'] = gt_names

                    if set_id in train_sets:
                        train_cal_infos.append(infos)
                    else:
                        test_cal_infos.append(infos)

    return train_cal_infos, test_cal_infos


def create_caltech_infos(root_path):

    train_sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
    test_sets = ['set06', 'set07', 'set08', 'set09', 'set10']

    print(f"train set: {len(train_sets)}, test set: {len(test_sets)}\n")

    print("Start Converting sequences to images:\n")
    _convert_seq_to_images(root_path)
    print("Finished Converting sequences to images:\n")

    train_cal_infos, test_cal_infos = _fill_trainval_infos(root_path, train_sets)

    print(f"train sample: {len(train_cal_infos)}, test sample: {len(test_cal_infos)}")

    with open(root_path + "/" + "infos_train.pkl", "wb") as f:
        pickle.dump(train_cal_infos, f)
    with open(root_path + "/" + "infos_test.pkl", "wb") as f:
        pickle.dump(test_cal_infos, f)
