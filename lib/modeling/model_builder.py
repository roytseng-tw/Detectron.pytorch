import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from modeling.roi_xfrom.roi_align.modules.roi_align import RoIAlign
from model.utils.net_utils import _affine_grid_gen
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import utils.blob as blob_utils

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
        # self.RPN = _RPN(self.Conv_Body.dim_out)
        self.RPN = rpn_heads.Single_Scale_RPN_Outputs(
            self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.POOLING_MODE == 'pool':
            self.Roi_Xform = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1 / 16)
        elif cfg.POOLING_MODE == 'crop':
            self.Roi_Xform = _RoICrop()
        elif cfg.POOLING_MODE == 'align':
            self.Roi_Xform = RoIAlign(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1 / 16, 0)
        else:
            raise ValueError(cfg.POOLING_MODE)

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(
                self.Mask_Head.dim_out, cfg.MODEL.NUM_CLASSES)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            # self.Keypoint_Outs = TODO

        self._init_modules()

        # Set trainning for all submodules. Must call after all submodules are added.
        self.train(train)

    def _init_modules(self):
        if self.Conv_Body.pretrained:
            state_dict = self.Conv_Body.get_pretrained_weights()
            self.Conv_Body.load_state_dict(
                {k: state_dict[k]
                 for k in self.Conv_Body.state_dict()})
        self.Conv_Body._init_modules()  # Freeze weights and reset bn running mean/var

        if self.Box_Head.pretrained:
            assert self.Conv_Body.pretrained
            self.Box_Head.load_state_dict(
                {k: state_dict[k]
                 for k in self.Box_Head.state_dict()})
        self.Box_Head._init_modules()

    def train(self, mode=True):
        # Override
        cfg.IS_TRAIN = mode
        super().train(mode)

    def forward(self, data, im_info, roidb=None,
                rpn_labels_int32_wide=None, rpn_bbox_targets_wide=None,
                rpn_bbox_inside_weights_wide=None, rpn_bbox_outside_weights_wide=None):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        batch_size = im_data.size(0)
        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)
        if not self.training:
            return_dict['blob_conv'] = blob_conv

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        rois = Variable(torch.from_numpy(rpn_ret['rois'])).cuda(device_id)
        return_dict['rois'] = rois
        if self.training:
            return_dict['rois_label'] = rpn_ret['labels_int32']

        rois_feat = self.roi_feature_transform(rois, blob_conv)

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(rois_feat)
            else:
                box_feat = self.Box_Head(rois_feat)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
        else:
            return_dict['rois_feat'] = rois_feat
            return return_dict  # TODO: complete the returns for RPN only situation

        if self.training:
            # rpn loss
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.single_scale_rpn_losses(
                rpn_ret['rpn_cls_logits'], rpn_ret['rpn_bbox_pred'],
                rpn_labels_int32_wide, rpn_bbox_targets_wide,
                rpn_bbox_inside_weights_wide, rpn_bbox_outside_weights_wide)
            return_dict['loss_rpn_cls'] = loss_rpn_cls
            return_dict['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['loss_rcnn_cls'] = loss_cls
            return_dict['loss_rcnn_bbox'] = loss_bbox

            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(rois_feat)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(
                    mask_pred, rpn_ret['masks_int32'])  #CHECK
                return_dict['loss_rcnn_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret['keypoint_weights'])
                else:
                    kps_feat = self.Keypoint_Head(rois_feat)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss TODO

        return return_dict

    def roi_feature_transform(self, rois, blob_conv):
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if cfg.POOLING_MODE == 'pool':
            rois_feat = self.Roi_Xform(blob_conv, rois)
        elif cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(
                rois,
                blob_conv.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            rois_feat = self.Roi_Xform(blob_conv, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                rois_feat = F.max_pool2d(rois_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            rois_feat = self.Roi_Xform(blob_conv, rois)
        return rois_feat

    def mask_net(self, mask_rois, blob_conv):
        """For inference"""
        if not self.training:
            mask_rois_feat = self.roi_feature_transform(mask_rois, blob_conv)
            mask_feat = self.Mask_Head(mask_rois_feat)
            mask_pred = self.Mask_Outs(mask_feat)
            return mask_pred

    def keypoint_net(self, keypoint_rois, blob_conv):
        """For inference"""
        if not self.training:
            keypoint_rois_feat = self.roi_feature_transform(
                keypoint_rois, blob_conv)
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
