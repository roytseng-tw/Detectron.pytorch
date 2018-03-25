from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
from modeling import resnet


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class mask_rcnn_outputs(nn.Module):
  def __init__(self, inplanes, n_classes):
    super().__init__()
    if not cfg.MRCNN.CLS_SPECIFIC_MASK:
      n_classes = 1
    self.classify = nn.Conv2d(inplanes, n_classes, 1, 1, 0)
    self._init_weights()

  def _init_weights(self):
    if cfg.MRCNN.CLS_SPECIFIC_MASK:
      weight_init_func = init.kaiming_normal
    else:
      weight_init_func = partial(init.normal, std=0.001)
    weight_init_func(self.classify.weight)
    init.constant(self.classify.bias, 0)

  def detectron_weight_mapping(self):
    mapping = {
      'classify.weight': 'mask_fcn_logits_w',
      'classify.bias': 'mask_fcn_logits_b'
    }
    orphan_in_detectron = []
    return mapping, orphan_in_detectron

  def forward(self, x):
    x = self.classify(x)
    if not self.training:
      x = F.sigmoid(x)
    return x


def mask_rcnn_losses(mask_pred, rois_mask, rois_label, weight):
  n_rois, n_classes, _, _ = mask_pred.size()
  rois_mask_label = rois_label[weight.data.nonzero().view(-1)]
  # select pred mask corresponding to gt label
  if cfg.MRCNN.MEMORY_EFFICIENT_LOSS:  # About 200~300 MB less. Not really sure how.
    mask_pred_select = Variable(mask_pred.data.new(n_rois, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION))
    for n, l in enumerate(rois_mask_label.data):
      mask_pred_select[n] = mask_pred[n, l]
  else:
    inds = rois_mask_label.data + \
      torch.arange(0, n_rois * n_classes, n_classes).long().cuda(rois_mask_label.data.get_device())
    mask_pred_select = mask_pred.view(-1, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION)[inds]
  loss = F.binary_cross_entropy_with_logits(mask_pred_select, rois_mask)
  return loss


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

class mask_rcnn_fcn_head_v0upshare(nn.Module):
  def __init__(self, inplanes):
    super().__init__()
    self.inplanes = inplanes
    self.dim_out = cfg.MRCNN.DIM_REDUCED
    self.SHARE_RES5 = True

    # the `self.res5` will be assigned later
    self.res5 = None
    _, dim_conv5 = ResNet_roi_conv5_head_for_masks(inplanes)
    self.upconv5 = nn.ConvTranspose2d(dim_conv5, self.dim_out, 2, 2, 0)

    self._init_weights()

  def _init_weights(self):
    if cfg.MRCNN.CONV_INIT == 'GaussianFill':
      init.normal(self.upconv5.weight, std=0.001)
    elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
      init.kaiming_normal(self.upconv5.weight)
    init.constant(self.upconv5.bias, 0)

  def share_res5_module(self, res5_target):
    self.res5 = res5_target

  def detectron_weight_mapping(self):
    detectron_weight_mapping, orphan_in_detectron = \
      resnet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
    # Assign None for res5 modules, indicating not care
    for k in detectron_weight_mapping:
      detectron_weight_mapping[k] = None

    detectron_weight_mapping.update({
      'upconv5.weight': 'conv5_mask_w',
      'upconv5.bias': 'conv5_mask_b'
    })
    return detectron_weight_mapping, orphan_in_detectron

  def forward(self, x, roi_mask_weights=None):
    if cfg.IS_TRAIN:
      # On training, we share the res5 computation with bbox head, so it's necessary to
      # sample 'useful' batches from the input x (res5_2_sum). 'Useful' means that the
      # batch (roi) has corresponding mask groundtruth, namely having positive roi_mask_weights.
      inds = roi_mask_weights.data.nonzero().view(-1)
      x = x[inds]
    else:
      # On testing, the computation is not shared with bbox head. This time the input `x`
      # only contains the useful roi batches, so we don't need roi_mask_weights for selectoin
      assert roi_mask_weights is None
      x = self.res5(x)
    x = self.upconv5(x)
    x = F.relu(x, inplace=True)
    return x


class mask_rcnn_fcn_head_v0up(nn.Module):
  def __init__(self, inplanes):
    super().__init__()
    self.inplanes = inplanes
    self.dim_out = cfg.MRCNN.DIM_REDUCED

    self.res5, dim_out = ResNet_roi_conv5_head_for_masks(inplanes)
    self.upconv5 = nn.ConvTranspose2d(dim_out, self.dim_out, 2, 2, 0)

    self._init_weights()

  def _init_weights(self):
    if cfg.MRCNN.CONV_INIT == 'GaussianFill':
      init.normal(self.upconv5.weight, std=0.001)
    elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
      init.kaiming_normal(self.upconv5.weight)
    init.constant(self.upconv5.bias, 0)

  def detectron_weight_mapping(self):
    detectron_weight_mapping, orphan_in_detectron = \
      resnet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
    detectron_weight_mapping.update({
      'upconv5.weight': 'conv5_mask_w',
      'upconv5.bias': 'conv5_mask_b'
    })
    return detectron_weight_mapping, orphan_in_detectron

  def forward(self, x):
    x = self.res5(x)
    # print(x.size()) e.g. (128, 2048, 7, 7)
    x = self.upconv5(x)
    x = F.relu(x, inplace=True)
    return x


def ResNet_roi_conv5_head_for_masks(dim_in):
  dilation = cfg.MRCNN.DILATION
  stride_init = cfg.POOLING_SIZE // 7  # by default: 2
  module, dim_out = resnet._make_layer(resnet.Bottleneck, dim_in, 512, 3, stride_init, dilation)
  return module, dim_out
