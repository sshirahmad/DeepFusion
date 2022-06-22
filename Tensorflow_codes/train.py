import tensorflow as tf2
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import numpy as np
import os
from time import time

from model.factory import model_factory

import config
from utils.logging import logger
from model.losses import BoxRegLoss, GridRegLoss, CrossEntropyLoss
from model.learning_rate_schedules import create_lr_scheduler, OneCycleScheduler
from utils.test_tools import bboxes_sort, bboxes_select, bboxes_nms_batch, match_boxes, compute_ap, \
    log_average_miss_rate
from utils.config import Config
from dataset.build_loader import create_data_loader
from evaluate import validation

tf.compat.v1.disable_eager_execution()

# define placeholders and variables
inputs = tf.placeholder(tf.float32, shape=(None, config.img_size, config.img_size, 3))
bboxes_gt1 = tf.placeholder(tf.float32, shape=(None, config.grid_size[0] * config.grid_size[0] * 3, 4))
bboxes_gt2 = tf.placeholder(tf.float32, shape=(None, config.grid_size[1] * config.grid_size[1] * 3, 4))
bboxes_gt3 = tf.placeholder(tf.float32, shape=(None, config.grid_size[2] * config.grid_size[2] * 3, 4))
label_gt1 = tf.placeholder(tf.int32, shape=(None, config.grid_size[0] * config.grid_size[0] * 3, 1))
label_gt2 = tf.placeholder(tf.int32, shape=(None, config.grid_size[1] * config.grid_size[1] * 3, 1))
label_gt3 = tf.placeholder(tf.int32, shape=(None, config.grid_size[2] * config.grid_size[2] * 3, 1))
grid_gt1 = tf.placeholder(tf.float32, shape=(None, config.grid_size[0] * config.grid_size[0], 1))
grid_gt2 = tf.placeholder(tf.float32, shape=(None, config.grid_size[1] * config.grid_size[1], 1))
grid_gt3 = tf.placeholder(tf.float32, shape=(None, config.grid_size[2] * config.grid_size[2], 1))
global_step = tf.Variable(0, trainable=False, name='global_step')
#sigma_gaussian1 = tf.Variable(-4., name="sigma_gaussian1", trainable=True)
#sigma_gaussian2 = tf.Variable(-4., name="sigma_gaussian2", trainable=True)
#sigma_gibbs = tf.Variable(-4., name="sigma_gibbs", trainable=True)
lr = tf.placeholder(dtype=tf.float32)


class Trainer:
    def __init__(self, config):
        self.neg_ratio = config.train.neg_ratio
        self.model_name = config.train.model_name
        self.backbone_name = config.train.backbone_name
        self.img_size = config.img_size

        self.log_step = config.logger.log_step
        self.summary_path = config.logger.summary_path
        self.summary_step = config.logger.summary_step

        self.save_step = config.checkpoint.save_step
        self.checkpoint_path = config.checkpoint.checkpoint_path

        self.det_func = BoxRegLoss(reduction='mean')
        self.clf_func = CrossEntropyLoss(reduction='mean')
        self.grid_func = GridRegLoss(reduction='mean')
        self.training_step = config.training_step
        self.learning_rate_scheduler = create_lr_scheduler(config.lr_scheduler, self.training_step)
        self.data_loader = create_data_loader(config.dataset)
        self.config = config
        self.validation = config.validation
        self.validation_step = config.checkpoint.save_step

        self.batch_size = config.dataset.batch_size
        self.test_config = config.test_config

    def hard_negative_mining(self, logits_pred, label_gt):
        pred = tf.nn.softmax(logits_pred)

        pos_mask = tf.reshape(label_gt, shape=[-1])
        pos_mask = tf.cast(pos_mask, dtype=tf.float32)

        neg_mask = tf.logical_not(tf.cast(pos_mask, dtype=tf.bool))
        neg_mask = tf.cast(neg_mask, dtype=tf.float32)

        # Hard negative mining...
        neg_score = tf.where(tf.cast(neg_mask, dtype=tf.bool),
                             pred[:, 0], 1. - neg_mask)

        # Number of negative entries to select.
        pos_num = tf.reduce_sum(pos_mask)
        max_neg_num = tf.cast(tf.reduce_sum(neg_mask), dtype=tf.int32)
        n_neg = tf.cast(self.neg_ratio * pos_num, tf.int32) + tf.shape(inputs)[0]
        n_neg = tf.minimum(n_neg, max_neg_num)

        val, _ = tf.nn.top_k(-neg_score, k=n_neg)
        max_hard_pred = -val[-1]

        nmask = tf.logical_and(tf.cast(neg_mask, dtype=tf.bool),
                               neg_score < max_hard_pred)
        hard_neg_mask = tf.cast(nmask, tf.float32)

        return hard_neg_mask, pos_mask, max_hard_pred

    def build_train_graph(self, is_training):

        net = model_factory(config=self.config, inputs=inputs, is_training=is_training)
        bboxes_pred1, logits_pred1, grid_pred1, bboxes_pred2, logits_pred2, grid_pred2, bboxes_pred3, logits_pred3, grid_pred3, = net.get_output_for_train()
        with tf.name_scope("clf_loss_process"):
            logits_pred1 = tf.reshape(logits_pred1, shape=[-1, 2])
            logits_pred2 = tf.reshape(logits_pred2, shape=[-1, 2])
            logits_pred3 = tf.reshape(logits_pred3, shape=[-1, 2])

            hard_neg_mask1, pos_mask1, max_hard_pred1 = self.hard_negative_mining(logits_pred1, label_gt1)
            hard_neg_mask2, pos_mask2, max_hard_pred2 = self.hard_negative_mining(logits_pred2, label_gt2)
            hard_neg_mask3, pos_mask3, max_hard_pred3 = self.hard_negative_mining(logits_pred3, label_gt3)

            tf.summary.scalar("max_hard_predition1", max_hard_pred1)  # the bigger, the better
            tf.summary.scalar("max_hard_predition2", max_hard_pred2)  # the bigger, the better
            tf.summary.scalar("max_hard_predition3", max_hard_pred3)  # the bigger, the better

            clf_loss1 = self.clf_func(logits_pred1, label_gt1, pos_mask1, hard_neg_mask1)
            clf_loss2 = self.clf_func(logits_pred2, label_gt2, pos_mask2, hard_neg_mask2)
            clf_loss3 = self.clf_func(logits_pred3, label_gt3, pos_mask3, hard_neg_mask3)

        with tf.name_scope("det_loss_process"):
            det_loss1 = self.det_func(bboxes_pred1, bboxes_gt1, pos_mask1)
            det_loss2 = self.det_func(bboxes_pred2, bboxes_gt2, pos_mask2)
            det_loss3 = self.det_func(bboxes_pred3, bboxes_gt3, pos_mask3)

        with tf.name_scope("grid_loss_process"):
            grid_loss1 = self.grid_func(grid_pred1, grid_gt1)
            grid_loss2 = self.grid_func(grid_pred2, grid_gt2)
            grid_loss3 = self.grid_func(grid_pred3, grid_gt3)

        det_loss = 4. * det_loss1 + 2. * det_loss2 + 1. * det_loss3
        clf_loss = 4. * clf_loss1 + 2. * clf_loss2 + 1. * clf_loss3
        grid_loss = 4. * grid_loss1 + 2. * grid_loss2 + 1. * grid_loss3

        return det_loss, clf_loss, grid_loss

    def build_optimizer(self, det_loss, clf_loss, grid_loss, var_list=None):
        with tf.name_scope("optimize"):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(reg_losses)

            loss = 5.0 * det_loss + 1.0 * grid_loss + 1.0 * clf_loss + reg_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(lr)
                if var_list == None:
                    train_ops = optimizer.minimize(loss, global_step=global_step)
                else:
                    train_ops = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

#            tf.summary.scalar('sigma_gaussian1', sigma_gaussian1)
#            tf.summary.scalar('sigma_gaussian2', sigma_gaussian2)
#            tf.summary.scalar('sigma_gibbs', sigma_gibbs)
            tf.summary.scalar("det_loss", det_loss)
            tf.summary.scalar("clf_loss", clf_loss)
            tf.summary.scalar("grid_loss", grid_loss)
            tf.summary.scalar("learning_rate", lr)

            return train_ops

    def run(self):
        # build graph
        logger.info('Building graph, using %s...' % (self.model_name))
        det_loss, clf_loss, grid_loss = self.build_train_graph(is_training=True)

        # build optimizer
        train_ops = self.build_optimizer(det_loss, clf_loss, grid_loss)

        # summary ops
        merge_ops = tf.summary.merge_all()
        logger.info('Build graph success...')
        logger.info('Total trainable parameters:%s' %
                    str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        current_step = 0
        with tf.Session(config=config) as sess:
            # create a summary writer
            summary_dir = os.path.join(self.summary_path)
            writer = tf.summary.FileWriter(summary_dir, sess.graph)

            # load weights from a checkpoint or initialize them
            self.load_checkpoint(ckpt, init, saver, sess)

            pd = self.data_loader(config=self.config.dataset, batch_size=self.batch_size, for_what='train',
                                  whether_aug=True)

            avg_det_loss = 0.
            avg_clf_loss = 0.
            avg_grid_loss = 0.
            avg_time = 0.
            while (True):
                learning_rate, _ = self.learning_rate_scheduler(current_step)

                start = time()
                imgs, tbox1, tbox2, tbox3, tcls1, tcls2, tcls3, tgrid1, tgrid2, tgrid3 = pd.load_batch()
                imgs = np.array(imgs)
                tcls1 = np.reshape(np.array(tcls1), newshape=[self.batch_size, -1, 1])
                tcls2 = np.reshape(np.array(tcls2), newshape=[self.batch_size, -1, 1])
                tcls3 = np.reshape(np.array(tcls3), newshape=[self.batch_size, -1, 1])
                t_box1 = np.reshape(np.array(tbox1), newshape=[self.batch_size, -1, 4])
                t_box2 = np.reshape(np.array(tbox2), newshape=[self.batch_size, -1, 4])
                t_box3 = np.reshape(np.array(tbox3), newshape=[self.batch_size, -1, 4])
                tgrid1 = np.reshape(np.array(tgrid1), newshape=[self.batch_size, -1, 1])
                tgrid2 = np.reshape(np.array(tgrid2), newshape=[self.batch_size, -1, 1])
                tgrid3 = np.reshape(np.array(tgrid3), newshape=[self.batch_size, -1, 1])

                t_ops, m_ops, current_step, d_loss, c_loss, g_loss = sess.run(
                    [train_ops, merge_ops, global_step, det_loss, clf_loss, grid_loss],
                    feed_dict={inputs: imgs, label_gt1: tcls1, bboxes_gt1: t_box1, grid_gt1: tgrid1,
                               label_gt2: tcls2, bboxes_gt2: t_box2, grid_gt2: tgrid2,
                               label_gt3: tcls3, bboxes_gt3: t_box3, grid_gt3: tgrid3,
                               lr: learning_rate})

                t = round(time() - start, 3)

                # write to the logger
                if self.log_step is not None:
                    # caculate average loss
                    step = current_step % self.log_step
                    avg_det_loss = (avg_det_loss * step + d_loss) / (step + 1.)
                    avg_clf_loss = (avg_clf_loss * step + c_loss) / (step + 1.)
                    avg_grid_loss = (avg_grid_loss * step + g_loss) / (step + 1.)
                    avg_time = (avg_time * step + t) / (step + 1.)
                    if current_step % self.log_step == self.log_step - 1:
                        # print info
                        logger.info('Step%s det_loss:%s clf_loss:%s grid_loss:%s time:%s' % (str(current_step),
                                                                                             str(avg_det_loss),
                                                                                             str(avg_clf_loss),
                                                                                             str(avg_grid_loss),
                                                                                             str(avg_time)))
                        avg_det_loss = 0.
                        avg_clf_loss = 0.
                        avg_grid_loss = 0.

                # write to tf logs
                if self.summary_step is not None:
                    if current_step % self.summary_step == self.summary_step - 1:
                        # summary
                        writer.add_summary(m_ops, current_step)

                # save weights
                self.save_checkpoint(current_step, saver, sess)

                # validation
                if self.validation:
                    if current_step % self.validation_step == self.validation_step - 1:
                        with tf.name_scope("validation"):
                            AP, f1, recall, precision, lamr = validation(self.config)

                            logger.info('AP:%s F1 Score:%s Recall:%s Precision:%s MR:%s' % (str(AP),
                                                                                            str(f1),
                                                                                            str(recall),
                                                                                            str(precision),
                                                                                            str(lamr)))

                if self.training_step is not None:
                    if current_step >= self.training_step:
                        logger.info('Exit training...')
                        break

    def load_checkpoint(self, ckpt, init, saver, sess):
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            sess.run(init)
            logger.info('TF variables init success...')

    def save_checkpoint(self, current_step, saver, sess):
        if self.save_step is not None:
            if current_step % self.save_step == self.save_step - 1:
                logger.info('Saving model...')
                model_name = os.path.join(self.checkpoint_path, self.model_name + '.model')
                saver.save(sess, model_name, global_step=current_step)
                logger.info('Save model sucess...')


def main(_):
    """
    start training
    """
    config_path = './config.py'
    cfg = Config.fromfile(config_path)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    tf.app.run()
