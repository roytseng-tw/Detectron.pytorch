import torch.nn as nn
import torch.nn.functional as F

from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss


class fast_rcnn_outputs(nn.Module):
  def __init__(self, dim_in, ):
    super().__init__()
    self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
      self.bbox_pred = nn.Linear(dim_in, 4)
    else:
      self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

  def detectron_weight_mapping(self):
    detectron_weight_mapping = {
      'cls_score.weight': 'cls_score_w',
      'cls_score.bias': 'cls_score_b',
      'bbox_pred.weight': 'bbox_pred_w',
      'bbox_pred.bias': 'bbox_pred_b'
    }
    orphan_in_detectron = []
    return detectron_weight_mapping, orphan_in_detectron

  def forward(self, x):
    x = x.squeeze(3).squeeze(2)
    cls_score = self.cls_score(x)
    if not self.training:
      cls_score = F.softmax(cls_score, dim=1)
    bbox_pred = self.bbox_pred(x)

    return cls_score, bbox_pred


def fast_rcnn_losses(cls_score, bbox_pred,
                     rois_label, rois_target, rois_inside_ws, rois_outside_ws):
  loss_cls = F.cross_entropy(cls_score, rois_label)
  loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
  return loss_cls, loss_bbox
