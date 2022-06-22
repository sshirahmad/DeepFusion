from __future__ import print_function
from torch import nn, Tensor
import math
from det3d.torchie.cnn import xavier_init, kaiming_init
from ..registry import BACKBONES
from ..utils import build_norm_layer


def batch_norm(num_features, eps=1e-05, momentum=0.1):
    norm_cfg = dict(type="BN", eps=eps, momentum=momentum)
    return build_norm_layer(norm_cfg, num_features)[1]


def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
        batch_norm(oup),
        nn.LeakyReLU(0.2),
    )


@BACKBONES.register_module
class VGG16(nn.Module):
    def __init__(self, num_input_features=3, width_mult=1.):
        super(VGG16, self).__init__()
        self.cfgs = [
            # k, c, s
            [3, 64, 1],
            [3, 128, 1],
            [3, 128, 1],
            [3, 256, 1],
            [3, 256, 1],
            [3, 256, 1],  # F1
            [3, 512, 1],
            [3, 512, 1],
            [3, 512, 1],  # F2
            [3, 512, 1],
            [3, 512, 1],
            [3, 512, 1],  # F3

        ]

        input_channel = 64
        layers = [conv_bn(num_input_features, input_channel, 3, 1)]
        # building inverted residual blocks
        for k, c, s in self.cfgs:
            output_channel = c
            layers.append(conv_bn(input_channel, output_channel, k, s))
            input_channel = output_channel
        self.convs = nn.Sequential(*layers)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x_conv = []
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if i == 1:
                x = self.pool(x)
            if i == 3:
                x = self.pool(x)
            if i == 6:
                x = self.pool(x)
                x_conv.append(x)
            if i == 9:
                x = self.pool(x)
                x_conv.append(x)
            if i == 12:
                x = self.pool(x)
                x_conv.append(x)

        multi_scale_features = {
            'layer6': x_conv[0],
            'layer9': x_conv[1],
            'layer12': x_conv[2],
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