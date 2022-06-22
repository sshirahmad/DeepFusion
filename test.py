from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import datetime
from PIL import Image

import numpy as np
import torch
from det3d.core.utils.yolo_utils import get_batch_statistics, compute_ap, xywh2xyxy, rescale_boxes
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    provide_determinism,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
import pickle
import time


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def open_pred(root):
    with open(os.path.join(root, "prediction.pkl"), "rb") as f:
        predictions = pickle.load(f)

    return predictions


def batch_processor(model, data, **kwargs):
    example = example_to_device(
        data, torch.cuda.current_device(), non_blocking=True
    )

    return model(example, return_loss=False, **kwargs)


def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                 "anno_box", "ind", "mask", 'cat']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "image",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points",
            "gt_boxes",
            "yolo_map1", "yolo_map2", "yolo_map3",
            "classifier_map1", "classifier_map2", "classifier_map3",
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking, dtype=torch.float)

        elif k in [
            "obj_mask1",
            "obj_mask2",
            "obj_mask3",
            "noobj_mask1",
            "noobj_mask2",
            "noobj_mask3"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking, dtype=torch.bool)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def visualize(precision_curve, recall_curve, output_dir, gt_annos, preds, matched_inds, image_paths, img_size):
    result_path = os.path.join(output_dir, "Results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = os.path.join(output_dir, "Results", "recall_precision.png")
    # Precision-Recall curve
    plt.figure(1)
    plt.plot(recall_curve, precision_curve)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid()
    plt.savefig(file_name)
    plt.close()

    img_detections = [None for _ in range(len(gt_annos))]  # Stores detections for each image index
    for i, (pred, ind) in enumerate(zip(preds, matched_inds)):
        img_detections[i] = pred[ind, :]

    image_path = os.path.join(output_dir, "DeepFusion")
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    print("\nSaving images...")
    # Iterate through images and save plot of detections
    for i, (im_det, gt, path) in enumerate(zip(img_detections, gt_annos, image_paths)):
        print("(%d) Image: '%s'" % (i, path))

        # Create plot
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        color_pred = colors[6]
        color_gt = colors[4]

        # Draw bounding boxes and labels of detections
        gt[:, 1:] = rescale_boxes(gt[:, 1:], img_size, img.shape[:2])
        for _, x1_gt, y1_gt, x2_gt, y2_gt in gt:
            w_gt = x2_gt - x1_gt
            h_gt = y2_gt - y1_gt

            bbox_gt = patches.Rectangle((x1_gt, y1_gt), w_gt, h_gt, linewidth=2, edgecolor=color_gt,
                                        facecolor="none")
            ax.add_patch(bbox_gt)

        if im_det is not None:
            # Rescale boxes to original image
            im_det[:, :4] = rescale_boxes(im_det[:, :4], img_size, img.shape[:2])
            for x1_pred, y1_pred, x2_pred, y2_pred, conf_score in im_det:
                w_pred = x2_pred - x1_pred
                h_pred = y2_pred - y1_pred

                # Create a Rectangle patch
                bbox_pred = patches.Rectangle((x1_pred, y1_pred), w_pred, h_pred, linewidth=2,
                                              edgecolor=color_pred, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox_pred)
                # Add label
                plt.text(
                    x1_pred,
                    y1_pred,
                    s=conf_score.numpy().round(2),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color_pred, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = "image %d" % i
        output_path = os.path.join(image_path, f"{filename}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    images = [img for img in os.listdir(image_path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(image_path, "video.avi"), fourcc, 2, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    cv2.destroyAllWindows()
    video.release()


def main():
    config = "../configs/mobilenetv3_yolov3_inria.py"
    checkpoint = "../tools/work_dirs/mobilenetv3_yolov3_inria/epoch_1200.pth"
    gpus = 1
    testset = True
    seed = None

    cfg = Config.fromfile(config)
    cfg.gpus = gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # set random seeds
    if seed is not None:
        logger.info("Set random seed to {}".format(seed))
        provide_determinism(seed)
    else:
        provide_determinism()

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

    if testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        img_size=cfg.img_dim,
        shuffle=False,
        pin_memory=True,
    )

    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    model = model.cuda()
    model.eval()

    logger.info(f"work dir: {cfg.work_dir}")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 / 3)

    time_start = 0
    time_end = 0

    prev_time = time.time()
    print("\nPerforming object detection:")
    labels = []
    sample_metrics = []
    sample_matched_inds = []
    gt = []
    preds = []
    image_paths = []
    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        targets = data_batch['gt_boxes']
        labels += targets[:, 1].tolist()
        targets[:, 1:] = xywh2xyxy(targets[:, 1:])
        targets[:, 1:] *= cfg.img_dim

        with torch.no_grad():
            outputs = batch_processor(model, data_batch, contain_pcd=cfg.contain_pcd)

        gt += [targets.numpy()]
        preds += outputs
        image_paths += list(data_batch['image_path'])
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\tBatch %d, Inference Time: %s" % (i, inference_time))

        sample_metric, sample_ind = get_batch_statistics(outputs, targets, iou_threshold=cfg.test_cfg.iou_thres)
        sample_metrics += sample_metric
        sample_matched_inds += sample_ind

    true_positives, pred_scores = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision_curve, recall_curve, precision, recall, AP, f1 = compute_ap(true_positives, pred_scores, labels)

    print("\n Total time per frame: ", (time_end - time_start) / (end - start))

    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    print(f"\nAverage Precision:{AP * 100}\n Recall:{recall * 100}\n Precision:{precision * 100}\n F1 score:{f1 * 100}\n")

    visualize(precision_curve, recall_curve, cfg.work_dir, gt, preds, sample_matched_inds, image_paths, cfg.img_dim)


if __name__ == "__main__":
    main()
