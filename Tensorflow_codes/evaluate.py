import tensorflow.compat.v1 as tf
import numpy as np
import os, time
from utils.test_tools import bboxes_sort, bboxes_select, fuse_scores, bboxes_nms_batch, match_boxes, compute_ap, \
    log_average_miss_rate
from utils.config import Config
from model.factory import model_factory
from matplotlib.ticker import NullLocator
from dataset.inria_person import provider as inria_person_pd
import utils.test_tools as test_tools
import matplotlib.patches as patches
from PIL import Image
from dataset.build_loader import create_data_loader
from utils.logging import logger
import config
import cv2
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# define placeholder
inputs = tf.placeholder(tf.float32, shape=(None, config.img_size, config.img_size, 3))


def build_graph(config, is_training):
    """build tf graph for predict
    Args:
        model_name: choose a model to build
        config_dict: some config for building net
        is_training: whether to train or test, here must be False
    Return:
        det_loss: a tensor with a shape [bs, priori_boxes_num, 4]
        clf_loss: a tensor with a shape [bs, priori_boxes_num, 2]
    """
    assert is_training == False
    net = model_factory(config=config, inputs=inputs, is_training=is_training)
    corner_bboxes, clf_pred, grid_map = net.get_output_for_test()

    # fscores = fuse_scores(clf_pred, corner_bboxes, grid_map, config.img_size, config.dataset.batch_size)
    score, bboxes = bboxes_select(clf_pred, corner_bboxes,
                                             select_threshold=config.test_config.conf_threshold)
    score, bboxes = bboxes_sort(score, bboxes)
    rscores, rbboxes = bboxes_nms_batch(score, bboxes,
                                                   nms_threshold=config.test_config.nms_threshold,
                                                   keep_top_k=config.test_config.keep_top_k)
    return rscores, rbboxes


def validation(cfg):
    provider = create_data_loader(cfg.dataset)
    scores, bboxes = build_graph(is_training=False, config=cfg)

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(cfg.checkpoint.checkpoint_path)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            raise ValueError("can not find checkpoint, pls check checkpoint_dir")

        pd = provider(config=cfg.dataset, for_what="evaluate", whether_aug=False)

        sample_metrics = []
        cmatched_boxes = []
        cmatched_scores = []
        gt_annos = []
        num_labels = 0
        while (True):
            norm_img, corner_bboxes_gt, img_path = pd.load_data_eval()
            if img_path is not None:
                scores_pred, bboxes_pred = sess.run([scores, bboxes],
                                                    feed_dict={inputs: np.array([norm_img])})
                try:
                    scores_pred = list(scores_pred.values())
                    bboxes_pred = list(bboxes_pred.values())

                    scores_pred = scores_pred[0][0]
                    bboxes_pred = bboxes_pred[0][0]

                    bboxes_pred[:, 0] = bboxes_pred[:, 0] * cfg.img_size
                    bboxes_pred[:, 1] = bboxes_pred[:, 1] * cfg.img_size
                    bboxes_pred[:, 2] = bboxes_pred[:, 2] * cfg.img_size
                    bboxes_pred[:, 3] = bboxes_pred[:, 3] * cfg.img_size
                    bboxes_pred = np.int32(bboxes_pred)
                except:
                    bboxes_pred = None
                    scores_pred = None

                idx = []
                for pred_i, pred_box in enumerate(bboxes_pred):
                    if pred_box.any() != 0:
                        idx.append(pred_i)
                pred_boxes_clean = bboxes_pred[idx]
                pred_scores_clean = scores_pred[idx]

                if len(idx) == 0:
                    pred_boxes_clean = None
                    pred_scores_clean = None

                target_boxes = np.int32(corner_bboxes_gt * cfg.img_size)
                gt_annos.append({'boxes': target_boxes, 'path': img_path})
                if pred_boxes_clean is not None:
                    batch_metrics, batch_boxes, batch_scores = match_boxes(pred_boxes_clean, pred_scores_clean,
                                                                           target_boxes,
                                                                           cfg.test_config.iou_threshold)

                    sample_metrics.append(batch_metrics)
                    num_labels += len(target_boxes.tolist())
                    if batch_boxes:
                        cmatched_boxes.append(np.array(batch_boxes))
                        cmatched_scores.append(np.array(batch_scores))
                    else:
                        cmatched_boxes.append(None)
                        cmatched_scores.append(None)
                else:
                    cmatched_boxes.append(None)
                    cmatched_scores.append(None)
                    num_labels += len(corner_bboxes_gt)

            else:
                break

        if len(sample_metrics) != 0:
            true_positives = []
            for x in sample_metrics:
                true_positives = np.concatenate([true_positives, x])
            precision_curve, recall_curve, precision, recall, AP, f1 = compute_ap(true_positives, num_labels)
        else:
            precision_curve = []
            recall_curve = []
            precision = 0
            recall = 0
            AP = 0
            f1 = 0

        lamr, mr, fppi = log_average_miss_rate(precision_curve, recall_curve)

    return AP * 100, f1 * 100, recall * 100, precision * 100, lamr * 100



def main(_):
    config_path = './config.py'
    cfg = Config.fromfile(config_path)
    provider = create_data_loader(cfg.dataset)
    scores, bboxes = build_graph(is_training=False, config=cfg)

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(cfg.checkpoint.checkpoint_path)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            raise ValueError("can not find checkpoint, pls check checkpoint_dir")

        pd = provider(config=cfg.dataset, for_what="evaluate", whether_aug=False)

        sample_metrics = []
        cmatched_boxes = []
        cmatched_scores = []
        gt_annos = []
        num_labels = 0
        while(True):
            norm_img, corner_bboxes_gt, img_path = pd.load_data_eval()
            if img_path is not None:
                scores_pred, bboxes_pred = sess.run([scores, bboxes],
                                                    feed_dict={inputs: np.array([norm_img])})
                try:
                    scores_pred = list(scores_pred.values())
                    bboxes_pred = list(bboxes_pred.values())

                    scores_pred = scores_pred[0][0]
                    bboxes_pred = bboxes_pred[0][0]

                    bboxes_pred[:, 0] = bboxes_pred[:, 0] * cfg.img_size
                    bboxes_pred[:, 1] = bboxes_pred[:, 1] * cfg.img_size
                    bboxes_pred[:, 2] = bboxes_pred[:, 2] * cfg.img_size
                    bboxes_pred[:, 3] = bboxes_pred[:, 3] * cfg.img_size
                    bboxes_pred = np.int32(bboxes_pred)
                except:
                    bboxes_pred = None
                    scores_pred = None

                idx = []
                for pred_i, pred_box in enumerate(bboxes_pred):
                    if pred_box.any() != 0:
                        idx.append(pred_i)
                pred_boxes_clean = bboxes_pred[idx]
                pred_scores_clean = scores_pred[idx]

                if len(idx) == 0:
                    pred_boxes_clean = None
                    pred_scores_clean = None

                target_boxes = np.int32(corner_bboxes_gt * cfg.img_size)
                gt_annos.append({'boxes': target_boxes, 'path': img_path})
                if pred_boxes_clean is not None:
                    batch_metrics, batch_boxes, batch_scores = match_boxes(pred_boxes_clean, pred_scores_clean, target_boxes,
                                                         cfg.test_config.iou_threshold)

                    sample_metrics.append(batch_metrics)
                    num_labels += len(target_boxes.tolist())
                    if batch_boxes:
                        cmatched_boxes.append(np.array(batch_boxes))
                        cmatched_scores.append(np.array(batch_scores))
                    else:
                        cmatched_boxes.append(None)
                        cmatched_scores.append(None)
                else:
                    cmatched_boxes.append(None)
                    cmatched_scores.append(None)
                    num_labels += len(corner_bboxes_gt)

            else:
                break

        if len(sample_metrics) != 0:
            true_positives = []
            for x in sample_metrics:
                true_positives = np.concatenate([true_positives, x])
            precision_curve, recall_curve, precision, recall, AP, f1 = compute_ap(true_positives, num_labels)
        else:
            precision_curve = []
            recall_curve = []
            precision = 0
            recall = 0
            AP = 0
            f1 = 0

        lamr, mr, fppi = log_average_miss_rate(precision_curve, recall_curve)
        logger.info('AP:%s F1 Score:%s Recall:%s Precision:%s MR:%s' % (str(AP * 100),
                                                                        str(f1 * 100),
                                                                        str(recall * 100),
                                                                        str(precision * 100),
                                                                        str(lamr * 100)))

        result_path = os.path.join(cfg.output_dir, "Results")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_name = os.path.join(cfg.output_dir, "Results", "recall_precision.png")
        # Precision-Recall curve
        plt.figure(1)
        plt.plot(recall_curve, precision_curve)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.grid()
        plt.savefig(file_name)
        plt.close()

        file_name = os.path.join(cfg.output_dir, "Results", "miss_fppi.png")
        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        plt.figure(2)
        plt.loglog(fppi, mr)
        plt.grid()
        plt.ylabel("Miss Rate")
        plt.xlabel("FPPI")
        plt.savefig(file_name)
        plt.close()

        img_detections = [None for _ in range(len(gt_annos))]  # Stores detections for each image index
        for i, (boxes, scores) in enumerate(zip(cmatched_boxes, cmatched_scores)):
            # Save image and detections
            if boxes is not None:
                img_detections[i] = np.concatenate([boxes, scores[..., np.newaxis]], axis=-1)

        image_path = os.path.join(cfg.output_dir, "DeepFusion")
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (im_det, gt) in enumerate(zip(img_detections, gt_annos)):
            path = gt["path"]

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
            img = cv2.resize(img, dsize=(cfg.img_size, cfg.img_size))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Bounding-box colors
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]
            color_pred = colors[6]
            color_gt = colors[4]

            # Draw bounding boxes and labels of detections
            minmax_boxes = gt["boxes"]
            for y1_gt, x1_gt, y2_gt, x2_gt in minmax_boxes:
                w_gt = x2_gt - x1_gt
                h_gt = y2_gt - y1_gt

                bbox_gt = patches.Rectangle((x1_gt, y1_gt), w_gt, h_gt, linewidth=2, edgecolor=color_gt,
                                            facecolor="none")
                ax.add_patch(bbox_gt)

            if im_det is not None:
                # Rescale boxes to original image
                for y1_pred, x1_pred, y2_pred, x2_pred, conf_score in im_det:
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
                        s=conf_score.round(2),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color_pred, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = os.path.basename(path).split(".")[0]
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


if __name__ == '__main__':
    tf.app.run()
