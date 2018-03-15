from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from model.utils.config import cfg
from model.mask_rcnn.mask_rcnn import _maskRCNN
from model.faster_rcnn.resnet import resnet50, resnet101, resnet152  # NOQA


class resnet(_maskRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.num_layers = num_layers
    self.model_path = 'data/pretrained_model/resnet%d_caffe.pth' % num_layers
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    super().__init__(classes, class_agnostic)

  def _init_modules(self):
    resnet = eval('resnet%d()' % self.num_layers)

    if self.pretrained:
      print("Loading pretrained weights from %s" % (self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad = False  # NOQA
    for p in self.RCNN_base[1].parameters(): p.requires_grad = False  # NOQA

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad = False  # NOQA
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad = False  # NOQA
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad = False  # NOQA

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False  # NOQA

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
