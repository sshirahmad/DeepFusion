from model.DeepFusion import DeepFusion

import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

model_map = {"DeepFusion": DeepFusion,
             }


class model_factory(object):
    def __init__(self, config, inputs, is_training):
        """init the model_factory
        Args:
            model_name: must be one of model_map

            inputs: a tensor with shape [bs, h, w, c]
            is_training: indicate whether to train or test.
        """
        assert config.train.model_name in model_map.keys()

        self.model_name = config.train.model_name
        self.config = config
        if config.train.model_name == 'DeepFusion':
            self.det_out1, self.clf_out1, self.grids1, self.det_out2, self.clf_out2, self.grids2, self.det_out3, self.clf_out3, self.grids3 \
                = model_map[config.train.model_name](inputs=inputs,
                                                     is_training=is_training, backbone_name=config.train.backbone_name,
                                                     bboxs_each_cell=len(config.dataset.anchors) // 3)
        else:
            raise ValueError('error')

    def get_output_for_train(self):
        """get the nets output
        Return:
            det_out: a tensor with a shape [bs, prioriboxes_num, 4], t_bboxes
            clf_out: a tensor with a shape [bs, prioriboxes_num, 2], without softmax
            grid_out: a tensor with a shape [bs, prioriboxes_num, 1], with sigmoid
        """
        return self.det_out1, self.clf_out1, self.grids1, self.det_out2, self.clf_out2, self.grids2, self.det_out3, self.clf_out3, self.grids3
        pass

    def _compute_grid_offsets(self, det_out, anchors, grid_size):
        y, x = np.mgrid[0:grid_size, 0:grid_size]
        x_center = (x.astype(np.float32) + 0.5) / np.float32(grid_size)
        y_center = (y.astype(np.float32) + 0.5) / np.float32(grid_size)
        h_pboxes = anchors[:, 0] / self.config.img_size
        w_pboxes = anchors[:, 1] / self.config.img_size
        y_c_pboxes = np.expand_dims(y_center, axis=-1)  # shape is (grid_h, grid_w, 1)
        x_c_pboxes = np.expand_dims(x_center, axis=-1)
        shape = tf.shape(det_out)
        det_out = tf.reshape(det_out, shape=[-1, grid_size, grid_size, len(self.config.dataset.anchors) // 3, 4])

        y_c_pb = []
        x_c_pb = []
        for i in range(len(anchors)):
            y_c_pb.append(y_c_pboxes)
            x_c_pb.append(x_c_pboxes)

        # shape is (1, grid_h. grid_w, n_pboxes)
        y_c_pb = tf.expand_dims(tf.concat(y_c_pb, axis=-1), axis=0)
        x_c_pb = tf.expand_dims(tf.concat(x_c_pb, axis=-1), axis=0)

        y_t = det_out[:, :, :, :, 0]  # shape is (bs, grid_h, grid_w, n_pboxes)
        x_t = det_out[:, :, :, :, 1]
        h_t = det_out[:, :, :, :, 2]
        w_t = det_out[:, :, :, :, 3]

        # center_bboxes encoded by [y_c, x_c, h, w]
        y_c = y_t * h_pboxes + y_c_pb
        x_c = x_t * w_pboxes + x_c_pb
        h = tf.exp(h_t) * h_pboxes
        w = tf.exp(w_t) * w_pboxes

        # conner_bboxes encoded by [ymin, xmin, ymax, xmax]
        ymin = y_c - h / 2.
        xmin = x_c - w / 2.
        ymax = y_c + h / 2.
        xmax = x_c + w / 2.

        corner_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        corner_bboxes = tf.reshape(corner_bboxes, shape=[shape[0], -1, 4])

        return corner_bboxes

    def get_output_for_test(self):
        """get the nets output
                Return:
                    det_out: a tensor with a shape [bs, prioriboxes_num, 4],
                             encoded by [ymin, xmin, ymax, xmax]
                    clf_out: a tensor with a shape [bs, prioriboxes_num, 2], after softmax
                """
        img_size = self.config.img_size
        valid_anchors = list(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        corner_bboxes1 = self._compute_grid_offsets(self.det_out1, self.config.dataset.anchors[valid_anchors[0]],
                                                    self.config.grid_size[0])
        clf_pred1 = tf.nn.softmax(self.clf_out1)
        grid_pred1 = tf.reshape(self.grids1, [-1, img_size // 8, img_size // 8, 1])

        corner_bboxes2 = self._compute_grid_offsets(self.det_out2, self.config.dataset.anchors[valid_anchors[1]],
                                                    self.config.grid_size[1])
        clf_pred2 = tf.nn.softmax(self.clf_out2)
        grid_pred2 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(tf.reshape(self.grids2, [-1, img_size // 16, img_size // 16, 1]))

        corner_bboxes3 = self._compute_grid_offsets(self.det_out3, self.config.dataset.anchors[valid_anchors[2]],
                                                    self.config.grid_size[2])
        clf_pred3 = tf.nn.softmax(self.clf_out3)
        grid_pred3 = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last')(tf.reshape(self.grids3, [-1, img_size // 32, img_size // 32, 1]))

        grid_cat = tf.concat([grid_pred1, grid_pred2, grid_pred3], axis=-1)
        grid_mean = tf.reduce_mean(grid_cat, axis=-1)

        corner_bboxes = tf.concat([corner_bboxes1, corner_bboxes2, corner_bboxes3], axis=1)
        clf_pred = tf.concat([clf_pred1, clf_pred2, clf_pred3], axis=1)

        return corner_bboxes, clf_pred, grid_mean


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    config_dict = {'backbone': 'mobilenet_v3'}
    net = model_factory(inputs=imgs, model_name="DeepFusion", is_training=True, config_dict=config_dict)
    corner_bboxes, clf_pred = net.get_output_for_test()
    print("Hi")
