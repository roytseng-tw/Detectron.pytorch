from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.dout = din if cfg.RPN.OUT_DIM_AS_IN_DIM else 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, self.dout, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios)
        if cfg.RPN.CLS_ACTIVATION == 'softmax':
            self.nc_score_out *= 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(self.dout, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(self.dout, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        if cfg.RPN.CLS_ACTIVATION == 'softmax':
            rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # 2(bg/fg)
            rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
            rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        else:
            rpn_cls_prob = F.sigmoid(rpn_cls_score)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        rpn_loss_cls = 0
        rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            if cfg.RPN.CLS_ACTIVATION == 'softmax':
                rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            else:
                rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(-1)
            rpn_label = rpn_data[0].view(-1)

            rpn_keep = Variable(rpn_label.ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            if cfg.RPN.CLS_ACTIVATION == 'softmax':
                rpn_label = Variable(rpn_label.long())
                rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            else:
                rpn_label = Variable(rpn_label)
                rpn_loss_cls = F.binary_cross_entropy_with_logits(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                           rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, rpn_loss_cls, rpn_loss_box
