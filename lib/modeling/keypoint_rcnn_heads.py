import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn


# ---------------------------------------------------------------------------- #
# Keypoint R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class keypoint_outputs(nn.Module):
    """Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    def __init__(self, dim_in):
        super().__init__()
        self.upsample_heatmap = (cfg.KRCNN.UP_SCALE > 1)

        if cfg.KRCNN.USE_DECONV:
            # Apply ConvTranspose to the feature representation; results in 2x # upsampling
            self.deconv = nn.ConvTranspose2d(
                dim_in, cfg.KRCNN.DECONV_DIM, cfg.KRCNN.DECONV_KERNEL,
                2, padding=int(cfg.KRCNN.DECONV_KERNEL / 2) - 1)
            dim_in = cfg.KRCNN.DECONV_DIM

        if cfg.KRCNN.USE_DECONV_OUTPUT:
            # Use ConvTranspose to predict heatmaps; results in 2x upsampling
            self.classify = nn.ConvTranspose2d(
                dim_in, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.DECONV_KERNEL,
                2, padding=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1))
        else:
            # Use Conv to predict heatmaps; does no upsampling
            self.classify = nn.Conv2d(dim_in, cfg.KRCNN.NUM_KEYPOINTS, 1, 1, padding=0)

        if self.upsample_heatmap:
            # self.upsample = nn.UpsamplingBilinear2d(scale_factor=cfg.KRCNN.UP_SCALE)
            self.upsample = mynn.BilinearInterpolation2d(
                cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.UP_SCALE)

        self._init_weights()

    def _init_weights(self):
        if cfg.KRCNN.USE_DECONV:
            init.normal_(self.deconv.weight, std=0.01)
            init.constant_(self.deconv.bias, 0)

        if cfg.KRCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.classify.weight, std=0.001)
        elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.classify.weight)
        else:
            raise ValueError(cfg.KRCNN.CONV_INIT)
        init.constant_(self.classify.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        if cfg.KRCNN.USE_DECONV:
            detectron_weight_mapping.update({
                'deconv.weight': 'kps_deconv_w',
                'deconv.bias': 'kps_deconv_b'
            })

        if self.upsample_heatmap:
            blob_name = 'kps_score_lowres'
            detectron_weight_mapping.update({
                'upsample.upconv.weight': None,  # 0: don't load from or save to checkpoint
                'upsample.upconv.bias': None
            })
        else:
            blob_name = 'kps_score'
        detectron_weight_mapping.update({
            'classify.weight': blob_name + '_w',
            'classify.bias': blob_name + '_b'
        })

        return detectron_weight_mapping, []

    def forward(self, x):
        if cfg.KRCNN.USE_DECONV:
            x = F.relu(self.deconv(x), inplace=True)
        x = self.classify(x)
        if self.upsample_heatmap:
            x = self.upsample(x)
        return x


def keypoint_losses(kps_pred, keypoint_locations_int32, keypoint_weights,
                    keypoint_loss_normalizer=None):
    """Mask R-CNN keypoint specific losses."""
    device_id = kps_pred.get_device()
    kps_target = Variable(torch.from_numpy(
        keypoint_locations_int32.astype('int64'))).cuda(device_id)
    keypoint_weights = Variable(torch.from_numpy(keypoint_weights)).cuda(device_id)
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    loss = F.cross_entropy(
        kps_pred.view(-1, cfg.KRCNN.HEATMAP_SIZE**2), kps_target, reduce=False)
    loss = torch.sum(loss * keypoint_weights) / torch.sum(keypoint_weights)
    loss *= cfg.KRCNN.LOSS_WEIGHT

    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        loss *= keypoint_loss_normalizer.item() # np.float32 to float
    return loss


# ---------------------------------------------------------------------------- #
# Keypoint heads
# ---------------------------------------------------------------------------- #

class roi_pose_head_v1convX(nn.Module):
    """Mask R-CNN keypoint head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
        kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        module_list = []
        for _ in range(cfg.KRCNN.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn = nn.Sequential(*module_list)
        self.dim_out = hidden_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.KRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(cfg.KRCNN.NUM_STACKED_CONVS):
            detectron_weight_mapping['conv_fcn.%d.weight' % (2*i)] = 'conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn.%d.bias' % (2*i)] = 'conv_fcn%d_b' % (i+1)

        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='keypoint_rois',
            method=cfg.KRCNN.ROI_XFORM_METHOD,
            resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return x
