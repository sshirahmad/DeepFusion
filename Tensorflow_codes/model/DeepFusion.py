import tensorflow.compat.v1 as tf
import tensorflow as tf2
from model.backbone.mobilenet_v2 import mobilenetv2
from model.backbone.Mobilenetv3 import mobilenetv3

weight_decay = 5e-4

extract_feature_names = {'mobilenet_v3': ['layer_7', 'layer_13', 'layer_17'],
                         'mobilenet_v2': ['layer_7', 'layer_14', 'layer_19'],
                         'mobilenet_v1': ['Conv2d_4_pointwise', 'Conv2d_8_pointwise', 'Conv2d_13_pointwise']}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hsigmoid(x, name='hsigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def hswish(x, name='hswish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def se_block(input_feature, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    Args:
        input_feature: a tensor with any shape.
        name: indicate the variable scope
    Return:
        a tensor after recalibration
    """

    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.constant_initializer(value=0.0)
    regularizer = tf.keras.regularizers.L2(weight_decay)

    with tf.variable_scope("se_" + name, reuse=tf.AUTO_REUSE):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.layers.average_pooling2d(input_feature, input_feature.get_shape()[1:-1], 1,
                                              padding='valid', data_format='channels_last', name='global_avg')

        excitation = tf.layers.dense(inputs=squeeze,
                                     units=_make_divisible(channel // ratio, 8),
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=regularizer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.nn.relu6(excitation)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=regularizer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        excitation = hsigmoid(excitation)
        scale = input_feature * excitation

    return scale


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            regularizer=tf.keras.regularizers.L2(weight_decay),
                            initializer=tf2.keras.initializers.GlorotUniform())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', data_format='NHWC')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def batch_norm(x, is_train=True, name='bn'):
    return tf.layers.batch_normalization(x, scale=True, training=is_train, name=name)


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, is_train=is_train, name='bn')
        net = hswish(net)
        return net


def grid_classifier(input, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, 1, 1, 1, 1, 1, bias=True, name='conv2d_linear')
        net = tf.nn.sigmoid(net)

        return net


def DeepFusion(inputs, is_training, backbone_name, bboxs_each_cell=3):
    """ the whole model is inspried by yolov2, what makes our model different is that
        our model use mobilenetV2 as backbone, and use different feature map to do a
        merge, and we add attention module to improve the performance.
    Args:
        inputs: a tensor with the shape of [batch_size, h, w, c], default should
                be [bs, 224, 224, 3], you can try different height and width
                with the input_check setting False, some height and width may
                cause error due to I use tf.space_to_depth to merge different features.
        bboxs_each_cell: describe the number of bboxs in each grib cell
        msf: indicate whether merge multi-scalar feature
        is_training: whether to train
    Return:
        det_out: a tensor with the shape[bs, N, 4], means [y_t, x_t, h_t, w_t]
        clf_out: a tensor with the shape[bs, N, 2], means [bg_score, obj_score]
    """
    assert backbone_name in ['mobilenet_v3', 'mobilenet_v2', 'mobilenet_v1']
    if backbone_name == 'mobilenet_v2':
        end_points = mobilenetv2(inputs=inputs, is_training=is_training)
    elif backbone_name == 'mobilenet_v3':
        end_points = mobilenetv3(inputs=inputs, is_training=is_training)
    else:
        raise NotImplementedError

    conv_channels = [[128, 64, 32, 16, 8, bboxs_each_cell * 6],
                     [256, 128, 64, 32, 16, bboxs_each_cell * 6],
                     [512, 256, 128, 64, 32, bboxs_each_cell * 6]]
    with tf.variable_scope('DeepFusion', reuse=tf.AUTO_REUSE):
        back_feat3 = end_points[extract_feature_names[backbone_name][2]]
        se_feat3 = se_block(back_feat3, name="block3")
        grids3 = grid_classifier(se_feat3, name='grid_classifier3')
        net3 = se_feat3 * tf.repeat(grids3, se_feat3.get_shape()[-1], axis=-1)
        for j, channel in enumerate(conv_channels[2]):
            if channel == bboxs_each_cell * 6:
                net3 = conv2d(net3, channel, 1, 1, 1, 1, bias=True, name='convlast3_1_%d' % j)
            else:
                net3 = conv2d_block(net3, channel, 1, 1, is_train=is_training, name='convlast3_1_%d' % j)
                if j == 2:
                    feats3 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(net3)
                net3 = conv2d_block(net3, channel, 3, 1, is_train=is_training, name='convlast3_3_%d' % j)

        net3 = tf.reshape(net3, shape=[tf.shape(inputs)[0], -1, 6])
        sz3 = tf.shape(net3)
        det_out3 = tf.slice(net3, begin=[0, 0, 0], size=[sz3[0], sz3[1], 4])  # [y_t, x_t, h_t, w_t]
        clf_out3 = tf.slice(net3, begin=[0, 0, 4], size=[sz3[0], sz3[1], 2])  # [bg_socre, obj_score]
        grid_out3 = tf.reshape(grids3, shape=[tf.shape(inputs)[0], -1, 1])

        back_feat2 = tf.concat([end_points[extract_feature_names[backbone_name][1]], feats3], axis=-1)
        se_feat2 = se_block(back_feat2, name="block2")
        grids2 = grid_classifier(se_feat2, name='grid_classifier2')
        net2 = se_feat2 * tf.repeat(grids2, se_feat2.get_shape()[-1], axis=-1)
        for j, channel in enumerate(conv_channels[1]):
            if channel == bboxs_each_cell * 6:
                net2 = conv2d(net2, channel, 1, 1, 1, 1, bias=True, name='convlast2_1_%d' % j)
            else:
                net2 = conv2d_block(net2, channel, 1, 1, is_train=is_training, name='convlast2_1_%d' % j)
                if j == 2:
                    feats2 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(net2)
                net2 = conv2d_block(net2, channel, 3, 1, is_train=is_training, name='convlast2_3_%d' % j)

        net2 = tf.reshape(net2, shape=[tf.shape(inputs)[0], -1, 6])
        sz2 = tf.shape(net2)
        det_out2 = tf.slice(net2, begin=[0, 0, 0], size=[sz2[0], sz2[1], 4])  # [y_t, x_t, h_t, w_t]
        clf_out2 = tf.slice(net2, begin=[0, 0, 4], size=[sz2[0], sz2[1], 2])  # [bg_socre, obj_score]
        grid_out2 = tf.reshape(grids2, shape=[tf.shape(inputs)[0], -1, 1])

        back_feat1 = tf.concat([end_points[extract_feature_names[backbone_name][0]], feats2], axis=-1)
        se_feat1 = se_block(back_feat1, name="block1")
        grids1 = grid_classifier(se_feat1, name='grid_classifier1')
        net1 = se_feat1 * tf.repeat(grids1, se_feat1.get_shape()[-1], axis=-1)
        for j, channel in enumerate(conv_channels[0]):
            if channel == bboxs_each_cell * 6:
                net1 = conv2d(net1, channel, 1, 1, 1, 1, bias=True, name='convlast1_1_%d' % j)
            else:
                net1 = conv2d_block(net1, channel, 1, 1, is_train=is_training, name='convlast1_1_%d' % j)
                net1 = conv2d_block(net1, channel, 3, 1, is_train=is_training, name='convlast1_3_%d' % j)

        net1 = tf.reshape(net1, shape=[tf.shape(inputs)[0], -1, 6])
        sz1 = tf.shape(net1)
        det_out1 = tf.slice(net1, begin=[0, 0, 0], size=[sz1[0], sz1[1], 4])  # [y_t, x_t, h_t, w_t]
        clf_out1 = tf.slice(net1, begin=[0, 0, 4], size=[sz1[0], sz1[1], 2])  # [bg_socre, obj_score]
        grid_out1 = tf.reshape(grids1, shape=[tf.shape(inputs)[0], -1, 1])

        return det_out1, clf_out1, grid_out1, det_out2, clf_out2, grid_out2, det_out3, clf_out3, grid_out3


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    backbone_name = 'mobilenet_v3'
    det_out1, clf_out1, grids1, det_out2, clf_out2, grids2, det_out3, clf_out3, grids3 = DeepFusion(inputs=imgs,
                                                                                                    is_training=True,
                                                                                                    backbone_name=backbone_name)
    pass
