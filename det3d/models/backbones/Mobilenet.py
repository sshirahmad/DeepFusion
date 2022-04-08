from torch import nn, Tensor
import torch
import math
from det3d.torchie.cnn import xavier_init, kaiming_init
from ..registry import BACKBONES
from ..utils import build_norm_layer


# regarding momentum in torch!
# X_{mov_avg} = X_{mov_avg} * (1-momentum) + X_{mean} * momentum
# regarding momentum in TF!
# X_{mov_avg} = X_{mov_avg} * momentum + X_{mean} * (1-momentum)


def batch_norm(num_features, eps=1e-5, momentum=0.1):
    norm_cfg = dict(type="BN", eps=eps, momentum=momentum)
    return build_norm_layer(norm_cfg, num_features)[1]


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


class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_max = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
        )

        self.fc_avg = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
        )

        self.activation = h_sigmoid()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3)

    def forward(self, x):
        b, c, h, w = x.size()

        # channel-wise attention
        yc_avg = self.avg_pool(x).view(b, c)
        yc_avg = self.fc_avg(yc_avg).view(b, c, 1, 1)
        yc_max = self.max_pool(x).view(b, c)
        yc_max = self.fc_max(yc_max).view(b, c, 1, 1)
        yc = self.activation(yc_max + yc_avg)
        channel_refined = x * yc

        # spatial attention
        ys_max = torch.max(x, dim=1, keepdim=True).values
        ys_avg = torch.mean(x, dim=1, keepdim=True)
        ys = torch.cat((ys_avg, ys_max), dim=1)
        ys = self.conv(ys)
        ys = self.activation(ys)

        return channel_refined * ys


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        batch_norm(oup),
        h_swish(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        batch_norm(oup),
        h_swish(),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                batch_norm(hidden_dim),
                h_swish() if use_hs else nn.ReLU(),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batch_norm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batch_norm(hidden_dim),
                h_swish() if use_hs else nn.ReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                batch_norm(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batch_norm(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV3Large(nn.Module):
    def __init__(self, num_input_features=3, width_mult=1.):
        super(MobileNetV3Large, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
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

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(num_input_features, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last layer
        self.conv = conv_1x1_bn(input_channel, exp_size)  # F3

        self._initialize_weights()

    def forward(self, x):
        x_inverted = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            x_inverted.append(x)

        x = self.conv(x)

        multi_scale_features = {
            'layer5': x_inverted[6],
            'layer11': x_inverted[12],
            'layer16': x,
        }

        return multi_scale_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m, nonlinearity='relu')  # TODO weight initialization could be changed
                # xavier_init(m)
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


@BACKBONES.register_module
class MobileNetV3Small(nn.Module):
    def __init__(self, num_input_features=3, width_mult=1.):
        super(MobileNetV3Small, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],  # F1
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],  # F2
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(num_input_features, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last layer
        self.conv = conv_1x1_bn(input_channel, exp_size)  # F3

        self._initialize_weights()

    def forward(self, x):
        x_inverted = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            x_inverted.append(x)

        x = self.conv(x)

        multi_scale_features = {
            'layer4': x_inverted[3],
            'layer9': x_inverted[8],
            'layer13': x,
        }

        return multi_scale_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m, nonlinearity='relu')  # TODO weight initialization could be changed
                # xavier_init(m)
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
