import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BilinearInterpolation2d(nn.Module):
    """Bilinear interpolation in space of scale.

    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(in_channels), range(out_channels), :, :] = bil_filt

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=self.up_scale, padding=self.padding)

        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)
