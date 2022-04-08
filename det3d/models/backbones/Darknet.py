from torch import nn, Tensor
import math
from det3d.torchie.cnn import xavier_init, kaiming_init
from ..registry import BACKBONES
from ..utils import build_norm_layer


def batch_norm(num_features, eps=1e-05, momentum=0.01):
    norm_cfg = dict(type="BN", eps=eps, momentum=momentum)
    return build_norm_layer(norm_cfg, num_features)[1]


class DarknetConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False):
        super(DarknetConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias
            ),
            batch_norm(out_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class DarknetResidual(nn.Module):
    def __init__(self, in_channel, filters1, filters2):
        super(DarknetResidual, self).__init__()
        self.res = nn.Sequential(
            DarknetConv(in_channel, filters1, 1, 1),
            DarknetConv(filters1, filters2, 3, 1)
        )

    def forward(self, x):
        return x + self.res(x)


@BACKBONES.register_module
class DarkNet(nn.Module):
    def __init__(self, num_input_features=3, width_mult=1.):
        super(DarkNet, self).__init__()

        layers = [DarknetConv(num_input_features, 32, kernel_size=3, stride=1),
                  DarknetConv(32, 64, kernel_size=3, stride=2)]

        # 1x residual blocks
        for i in range(1):
            layers.append(DarknetResidual(64, 32, 64))

        layers.append(DarknetConv(64, 128, kernel_size=3, stride=2))
        # 2x residual blocks
        for i in range(2):
            layers.append(DarknetResidual(128, 64, 128))

        layers.append(DarknetConv(128, 256, kernel_size=3, stride=2))
        # 8x residual blocks
        for i in range(8):
            layers.append(DarknetResidual(256, 128, 256))

        self.features1 = nn.Sequential(*layers)

        layers = [DarknetConv(256, 512, kernel_size=3, stride=2)]
        # 8x residual blocks
        for i in range(8):
            layers.append(DarknetResidual(512, 256, 512))

        self.features2 = nn.Sequential(*layers)

        layers = [DarknetConv(512, 1024, kernel_size=3, stride=2)]
        # 4x residual blocks
        for i in range(4):
            layers.append(DarknetResidual(1024, 512, 1024))

        self.features3 = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)

        multi_scale_features = {
            'layer5': x1,
            'layer11': x2,
            'layer16': x3,
        }

        return multi_scale_features
