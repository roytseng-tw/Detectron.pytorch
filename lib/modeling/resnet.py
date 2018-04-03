import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn as mynn
from core.config import cfg
from utils.resnet_weights_helper import convert_state_dict

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #


def ResNet50_Conv4_Body():
    return ResNet_ConvX_Body((3, 4, 6))


def ResNet50_Conv5_Body():
    return ResNet_ConvX_Body((3, 4, 6, 3))


def ResNet101_Conv4_Body():
    return ResNet_ConvX_Body((3, 4, 23))


def ResNet101_Conv5_Body():
    return ResNet_ConvX_Body((3, 4, 23, 3))


def ResNet152_Conv5_Body():
    return ResNet_ConvX_Body((3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


class ResNet_ConvX_Body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                         ('bn1', mynn.AffineChannel2d(64)),
                         ('relu', nn.ReLU(inplace=True)),
                         ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        dim_in = 64
        dim_bottleneck = 64
        self.res2, dim_in = _make_layer(Bottleneck, dim_in, dim_bottleneck,
                                        block_counts[0])
        self.res3, dim_in = _make_layer(
            Bottleneck, dim_in, dim_bottleneck * 2, block_counts[1], stride=2)
        self.res4, dim_in = _make_layer(
            Bottleneck, dim_in, dim_bottleneck * 4, block_counts[2], stride=2)
        if len(block_counts) == 4:
            if cfg.RESNETS.RES5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.res5, dim_in = _make_layer(
                Bottleneck, dim_in, dim_bottleneck * 8, block_counts[3],
                stride, cfg.RESNETS.RES5_DILATION)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'res1.conv1.weight': 'conv1_w',
            'res1.bn1.weight': 'res_conv1_bn_s',
            'res1.bn1.bias': 'res_conv1_bn_b',
        }
        orphan_in_detectron = ['conv1_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


class ResNet_Roi_Conv5_Head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        stride_init = cfg.POOLING_SIZE // 7
        self.res5, self.dim_out = _make_layer(Bottleneck, dim_in, 512, 3,
                                              stride_init)
        assert self.dim_out == 2048
        self.avgpool = nn.AvgPool2d(7)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
          residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


# ---------------------------------------------------------------------------- #
# Helper functions and components
# ---------------------------------------------------------------------------- #


def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + '_bn_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + '_bn_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.bn%d.weight' % i] = dtt_bp + '_bn_s'
            mapping_to_detectron[my_prefix
                                 + '.bn%d.bias' % i] = dtt_bp + '_bn_b'

    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


def _make_layer(block, inplanes, planes, blocks, stride=1, dilation=1):
    """Make a number of [blocks] residual blocks.
    Returns:
        - a sequentail module of residual blocks
        - final output dimension
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            mynn.AffineChannel2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes, planes, stride, dilation=dilation,
            downsample=downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers), inplanes


class Bottleneck(nn.Module):
    """ Bottleneck Residual Block """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1,
                                                                       stride)

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = mynn.AffineChannel2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=str3x3,
            padding=1 * dilation,
            dilation=dilation,
            bias=False)
        self.bn2 = mynn.AffineChannel2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
