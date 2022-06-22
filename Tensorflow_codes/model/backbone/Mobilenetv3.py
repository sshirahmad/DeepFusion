import tensorflow.compat.v1 as tf
import tensorflow as tf2
import collections

weight_decay = 5e-4


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


def relu(x, name='relu'):
    return tf.nn.relu(x, name)


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

    with tf.variable_scope("se_" + name):
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


def batch_norm(x, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                                         scale=True,
                                         training=train,
                                         name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            regularizer=tf.keras.regularizers.L2(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', data_format='NHWC')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = hswish(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        conv = conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)
        return conv


def dwise_conv(input, k=3, channel_multiplier=1, strides=[1, 1, 1, 1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k, k, in_channel, channel_multiplier],
                            regularizer=tf.keras.regularizers.L2(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel * channel_multiplier],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, bottleneck_dim, output_dim, k, stride, is_train, se, hs, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input_dim = input.get_shape().as_list()[-1]
        if input_dim == bottleneck_dim:
            # dw
            net = dwise_conv(input, k=k, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = batch_norm(net, train=is_train, name='dw_bn')
            if se:
                net = se_block(net, 'se_block', ratio=8)
            net = hswish(net) if hs else relu(net)
            # pw & linear
            net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = batch_norm(net, train=is_train, name='pw_linear_bn')
        else:
            # pw
            net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
            net = batch_norm(net, train=is_train, name='pw_bn')
            net = hswish(net) if hs else relu(net)
            # dw
            net = dwise_conv(net, k=k, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = batch_norm(net, train=is_train, name='dw_bn')
            if se:
                net = se_block(net, 'se_block', ratio=8)
            net = hswish(net) if hs else relu(net)
            # pw & linear
            net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if stride == 1 and input_dim == output_dim:
            net = input + net

        return net


def mobilenetv3(inputs, is_training, width_mult=1.0):
    """mobilenetv2( deleted the global average pooling )
    Args:
        inputs: a tensor with the shape (bs, h, w, c)
        is_training: indicate whether to train or test
    Return:
        all the end point.
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],  # F1
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],  # F2
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    endPoints = collections.OrderedDict()
    with tf.variable_scope('mobilenetv3'):
        input_channel = _make_divisible(16 * width_mult, 8)
        net = conv2d_block(inputs, input_channel, 3, 2, is_training, name='conv1_1')  # size/2
        endPoints['layer_1'] = net

        for i, (k, t, c, use_se, use_hs, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            net = res_block(net, exp_size, output_channel, k, s, is_training, use_se, use_hs, name='res_{}'.format(i))
            endPoints['layer_{}'.format(i + 2)] = net
            input_channel = output_channel

        net = conv2d_block(net, exp_size, 1, 1, is_training, name='conv17_1')
        endPoints['layer_17'] = net

    return endPoints
