import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade_v2 import _ProposalTargetLayer
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.utils.net_utils import _affine_grid_gen
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
      return None
    try:
      parts = func_name.split('.')
      # Refers to a function in this module
      if len(parts) == 1:
        return globals()[parts[0]]
      # Otherwise, assume we're referencing a module under modeling
      module_name = 'modeling.' + '.'.join(parts[:-1])
      module = importlib.import_module(module_name)
      return getattr(module, parts[-1])
    except Exception:
      logger.error('Failed to find function: {}'.format(func_name))
      raise


class Generalized_RCNN(nn.Module):
  def __init__(self, train=False):
    super().__init__()
    self.training = train
    cfg.IS_TRAIN = train  # Be sure to use this value on module init

    # Backbone for feature extraction
    self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

    # Region Proposal Network
    self.RPN = _RPN(self.Conv_Body.dim_out)
    if self.training:
      self.Proposal_Target = _ProposalTargetLayer(cfg.MODEL.NUM_CLASSES)
    if cfg.POOLING_MODE == 'pool':
      self.Roi_Xform = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1 / 16)
    elif cfg.POOLING_MODE == 'crop':
      self.Roi_Xform = _RoICrop()
    elif cfg.POOLING_MODE == 'align':
      self.Roi_Xform = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1 / 16)
    else:
      raise ValueError(cfg.POOLING_MODE)

    # BBOX Branch
    if not cfg.MODEL.RPN_ONLY:
      self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(self.RPN.dout)
      self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

    # Mask Branch
    if cfg.MODEL.MASK_ON:
      self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(self.RPN.dout)
      if getattr(self.Mask_Head, 'SHARE_RES5', False):
        self.Mask_Head.share_res5_module(self.Box_Head.res5)
      self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out, cfg.MODEL.NUM_CLASSES)

    # Keypoints Branch
    if cfg.MODEL.KEYPOINTS_ON:
      self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(self.RPN.dout)
      if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
        self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
      # self.Keypoint_Outs = TODO

    self._init_modules()

    # Set trainning for all submodules. Must call at last.
    self.train(train)

  def _init_modules(self):
    if self.Conv_Body.pretrained:
      state_dict = self.Conv_Body.get_pretrained_weights()
      self.Conv_Body.load_state_dict({k: state_dict[k] for k in self.Conv_Body.state_dict()})
    self.Conv_Body._init_modules()  # Freeze weights and reset bn running mean/var

    if self.Box_Head.pretrained:
      assert self.Conv_Body.pretrained
      self.Box_Head.load_state_dict({k: state_dict[k] for k in self.Box_Head.state_dict()})
    self.Box_Head._init_modules()

  def train(self, mode=True):
    # Override
    cfg.IS_TRAIN = mode
    super().train(mode)

  def forward(self, *inputs):
    # Parse inputs
    inputs = list(inputs)
    im_data = inputs.pop(0)
    im_info = inputs.pop(0).data
    gt_boxes = inputs.pop(0).data
    num_boxes = inputs.pop(0).data
    if self.training:
      if cfg.MODEL.MASK_ON:
        gt_masks = inputs.pop(0).data.cpu()
      else:
        gt_masks = None
      if cfg.MODEL.KEYPOINTS_ON:
        gt_poses = inputs.pop(0).data.cpu()
      else:
        gt_poses = None
    BATCH_SIZE = im_data.size(0)

    return_dict = {}  # A dict to collect return variables

    blob_conv = self.Conv_Body(im_data)
    if not self.training:
      return_dict['blob_conv'] = blob_conv

    # rois:
    #   shape: [batch, n_rois, 5], n_rois < cfg.TRAIN.RPN_POST_NMS_TOP_N
    #   (n, x1, y1, x2, y2) specifying an image batch index n and a rectangle (x1, y1, x2, y2).
    #   Rectangles are inside the image.
    rois, rpn_loss_cls, rpn_loss_bbox = self.RPN(blob_conv, im_info, gt_boxes, num_boxes)
    NUM_ROIS = rois.size(1)

    if cfg.IS_TRAIN:
      rois_data = self.Proposal_Target(rois, gt_boxes, gt_masks, gt_poses)
      rois_data = list(rois_data)
      rois = rois_data.pop(0)
      rois_data = list(map(Variable, rois_data))

      rois_label = rois_data.pop(0).view(-1).long()
      rois_target = rois_data.pop(0).view(-1, rois_target.size(2))
      rois_inside_ws = rois_data.pop(0).view(-1, rois_inside_ws.size(2))
      rois_outside_ws = rois_data.pop(0).view(-1, rois_outside_ws.size(2))
      if cfg.MODEL.MASK_ON:
        rois_mask = rois_data.pop(0).view(-1, rois_mask.size(2), rois_mask.size(3))
        rois_mask_ws = rois_data.pop(0).view(-1)
      if cfg.MODEL.KEYPOINTS_ON:
        rois_pose = rois_data.pop(0).view(-1)
        rois_pose_ws = rois_data.pop(0).view(-1)
      return_dict['rois_label'] = rois_label

    rois = Variable(rois)
    return_dict['rois'] = rois

    rois_feat = self.roi_feature_transform(rois, blob_conv)

    if not cfg.MODEL.RPN_ONLY:
      if cfg.MODEL.SHARE_RES5 and self.training:
        box_feat, res5_feat = self.Box_Head(rois_feat)
      else:
        box_feat = self.Box_Head(rois_feat)
      cls_score, bbox_pred = self.Box_Outs(box_feat)
      return_dict['cls_score'] = cls_score.view(BATCH_SIZE, NUM_ROIS, -1)
      return_dict['bbox_pred'] = bbox_pred.view(BATCH_SIZE, NUM_ROIS, -1)
    else:
      return_dict['rois_feat'] = rois_feat
      return return_dict  # TODO: complete the returns for RPN only situation

    if self.training:

      if cfg.MODEL.MASK_ON:
        if getattr(self.Mask_Head, 'SHARE_RES5', False):
          mask_feat = self.Mask_Head(res5_feat, rois_mask_ws)
        else:
          mask_feat = self.Mask_Head(rois_feat)
        mask_pred = self.Mask_Outs(mask_feat)
        return_dict['mask_pred'] = mask_pred.view(BATCH_SIZE, NUM_ROIS,
                                                  cfg.MRCNN.RESOLUTION,
                                                  cfg.MRCNN.RESOLUTION)

      if cfg.MODEL.KEYPOINTS_ON:
        if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
          kps_feat = self.Keypoint_Head(res5_feat, rois_pose_ws)
        else:
          kps_feat = self.Keypoint_Head(rois_feat)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return_dict['keypoints_pred'] = kps_pred.view(BATCH_SIZE, NUM_ROIS,
                                                      cfg.KRCNN.HEATMAP_SIZE,
                                                      cfg.KRCNN.HEATMAP_SIZE)

      # calculate loss
      loss_cls, loss_bbox = fast_rcnn_heads.fast_rcnn_losses(
        cls_score, bbox_pred, rois_label, rois_target, rois_inside_ws, rois_outside_ws)
      return_dict['loss_cls'] = loss_cls
      return_dict['loss_bbox'] = loss_bbox

      if cfg.MODEL.MASK_ON:
        loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rois_mask, rois_label, rois_mask_ws) #TODO
        return_dict['loss_mask'] = loss_mask

      # if cfg.MODEL.KEYPOINTS_ON:
      #   loss_keypoints =

    return return_dict

  def roi_feature_transform(self, rois, blob_conv):
    if cfg.POOLING_MODE == 'pool':
        rois_feat = self.Roi_Xform(blob_conv, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'crop':
        grid_xy = _affine_grid_gen(rois.view(-1, 5), blob_conv.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
        rois_feat = self.Roi_Xform(blob_conv, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            rois_feat = F.max_pool2d(rois_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        rois_feat = self.Roi_Xform(blob_conv, rois.view(-1, 5))
    return rois_feat

  def mask_net(self, mask_rois, blob_conv):
    """For inference
    """
    if not self.training:
      mask_rois_feat = self.roi_feature_transform(mask_rois, blob_conv)
      mask_feat = self.Mask_Head(mask_rois_feat)
      mask_pred = self.Mask_Outs(mask_feat)
      return mask_pred

  def keypoint_net(self, keypoint_rois, blob_conv):
    """For inference
    """
    if not self.training:
      keypoint_rois_feat = self.roi_feature_transform(keypoint_rois, blob_conv)
      kps_feat = self.Keypoint_Head(keypoint_rois_feat)
      kps_pred = self.Keypoint_Outs(kps_feat)
      return kps_pred

  def detectron_weight_mapping(self):
    d_wmap = {}  # detectron_weight_mapping
    d_orphan = []  # detectron orphan weight list
    for name, m_child in self.named_children():
      if list(m_child.parameters()):  # if module has any parameter
        child_map, child_orphan = m_child.detectron_weight_mapping()
        d_orphan.extend(child_orphan)
        for key, value in child_map.items():
          new_key = name + '.' + key
          d_wmap[new_key] = value

    return d_wmap, d_orphan
