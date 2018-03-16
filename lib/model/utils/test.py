# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

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
import model.utils.boxes as box_utils


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

  [[im_h, im_w, im_scale]] = im_info.data.cpu().numpy()
  im_h, im_w = im_h.astype(int), im_w.astype(int)
  if mask_pred is not None:
    mask_pred = mask_pred.data.cpu().numpy()
    timers['im_detect_mask'].tic()
    masks = im_test_mask(mask_pred)
    timers['im_detect_mask'].toc()
    timers['misc_mask'].tic()
    cls_segms = segm_results(cls_boxes, masks, boxes, im_h, im_w)
    timers['misc_mask'].toc()
  else:
    cls_segms = None

  if pose_pred is not None:
    pass
  else:
    cls_keyps = None

  return cls_boxes, cls_segms, cls_keyps


def im_test_bbox(im_info, rois, cls_prob, bbox_pred, args):
  """Prepare the bbox for testing
  """
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


def im_test_mask(mask_pred):
  """Prepare the mask for testing
  Returns:
      pred_masks (ndarray): R x K x M x M array of class specific soft masks
          output by the network (must be processed by segm_results to convert
          into hard masks in the original image coordinate space)
  """
  M = cfg.MRCNN.RESOLUTION

  if cfg.MRCNN.CLS_SPECIFIC_MASK:
    pred_masks = mask_pred.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
  else:
    pred_masks = mask_pred.reshape([-1, 1, M, M])

  # # truncate padded masks ??
  # print(pred_masks.shape)
  # nonzeros = np.any(pred_masks, axis=(1,2,3))
  # print(nonzeros.shape)
  # print(np.count_nonzero(nonzeros))
  # exit()

  return pred_masks


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
  num_classes = cfg.MODEL.NUM_CLASSES
  cls_segms = [[] for _ in range(num_classes)]
  mask_ind = 0
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.
  M = cfg.MRCNN.RESOLUTION
  scale = (M + 2.0) / M
  ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

  # skip j = 0, because it's the background class
  for j in range(1, num_classes):
    segms = []
    for _ in range(cls_boxes[j].shape[0]):
      if cfg.MRCNN.CLS_SPECIFIC_MASK:
        padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
      else:
        padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

      ref_box = ref_boxes[mask_ind, :]
      w = ref_box[2] - ref_box[0] + 1
      h = ref_box[3] - ref_box[1] + 1
      w = np.maximum(w, 1)
      h = np.maximum(h, 1)

      mask = cv2.resize(padded_mask, (w, h))
      mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
      im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

      x_0 = max(ref_box[0], 0)
      x_1 = min(ref_box[2] + 1, im_w)
      y_0 = max(ref_box[1], 0)
      y_1 = min(ref_box[3] + 1, im_h)

      im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
        (x_0 - ref_box[0]):(x_1 - ref_box[0])
      ]

      # Get RLE encoding used by the COCO evaluation API
      rle = mask_util.encode(
        np.array(im_mask[:, :, np.newaxis], order='F')
      )[0]
      segms.append(rle)

      mask_ind += 1

    cls_segms[j] = segms

  # assert mask_ind == masks.shape[0], '{}, {}'.format(mask_ind, masks.shape[0])
  return cls_segms