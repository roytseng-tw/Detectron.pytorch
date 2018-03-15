from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms_wrapper import nms
from model.utils.config import cfg
from model.utils.timer import Timer


def im_test_all(
  args, im_info, rois, rois_label, cls_prob, bbox_pred, mask_pred=None, pose_pred=None,
  timers=None):
  """Process the outputs of model for testing
  args: arguments from command line.
  timer: record the cost of time for different steps
  The rest of inputs are of type pytorch Variables and either input to or output from the model.
  """

  assert len(im_info) == 1, "Only single-image / single-scale batch implemented"

  if timers is None:
    timers = defaultdict(Timer)

  scores, boxes = im_test_bbox(im_info, rois, cls_prob, bbox_pred, args)
  scores = scores.view(-1, scores.size(-1))
  boxes = boxes.view(-1, boxes.size(-1))

  # score and boxes are from the whole image after score thresholding and nms
  # (they are not separated by class)
  # cls_boxes boxes and scores are separated by class and in the format used
  # for evaluating results
  timers['misc_bbox'].tic()
  scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes, args)
  timers['misc_bbox'].toc()

  # return scores, boxes
  return cls_boxes   #, cls_segms, cls_keyps


def im_test_bbox(im_info, rois, cls_prob, bbox_pred, args):
  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]
  num_classes = scores.size(-1)

  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred.data
    BATCH_SIZE = bbox_pred.size(0)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Optionally normalize targets by a precomputed mean and stdev
      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
      if args.class_agnostic:
        box_deltas = box_deltas.view(BATCH_SIZE, -1, 4)
      else:
        box_deltas = box_deltas.view(BATCH_SIZE, -1, 4 * num_classes)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, BATCH_SIZE, weights=cfg.MODEL.BBOX_REG_WEIGHTS)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, BATCH_SIZE)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))  #FIXME: convert to torch.Tensor

  # unscale back to raw image space
  pred_boxes /= im_info.data[:, 2].view(-1, 1, 1).expand(1, pred_boxes.size(1), 1)

  return scores, pred_boxes


def box_results_with_nms_and_limit(scores, boxes, args):
  """Returns bounding-box detection results by thresholding on scores and
  applying non-maximum suppression (NMS).

  `boxes` has shape (#detections, 4 * #classes), where each row represents
  a list of predicted bounding boxes for each of the object classes in the
  dataset (including the background class). The detections in each row
  originate from the same object proposal.

  `scores` has shape (#detection, #classes), where each row represents a list
  of object detection confidence scores for each of the object classes in the
  dataset (including the background class). `scores[i, j]`` corresponds to the
  box at `boxes[i, j * 4:(j + 1) * 4]`.
  """
  num_classes = scores.size(-1)
  cls_boxes = [[] for _ in range(num_classes)]
  empty_array = np.array([[], [], [], [], []]).T
  # Apply threshold on detection probabilities and apply NMS
  # Skip j = 0, because it's the background class
  for j in range(1, num_classes):
      inds = torch.nonzero(scores[:, j] > cfg.TEST.SCORE_THRESH).view(-1)
      if inds.numel() > 0:
        scores_j = scores[:, j][inds]
        if args.class_agnostic:
          boxes_j = boxes[inds, :]
        else:
          boxes_j = boxes[inds][:, j * 4:(j + 1) * 4]
        dets_j = torch.cat((boxes_j, scores_j.unsqueeze(1)), 1)
        keep = nms(dets_j, cfg.TEST.NMS).view(-1).long()
        nms_dets = dets_j[keep]
        cls_boxes[j] = nms_dets.cpu().numpy()
      else:
        cls_boxes[j] = empty_array

  # Limit to max_per_image detections **over all classes**
  if cfg.TEST.DETECTIONS_PER_IM > 0:
    image_scores = np.hstack(
        [cls_boxes[j][:, -1] for j in range(1, num_classes)]
    )
    if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
        image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
        for j in range(1, num_classes):
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]

  im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
  boxes = im_results[:, :-1]
  scores = im_results[:, -1]
  return scores, boxes, cls_boxes
