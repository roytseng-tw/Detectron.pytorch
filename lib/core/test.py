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

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.keypoints as keypoint_utils


def im_detect_all(model, im, box_proposals=None, timers=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        raise NotImplementedError
        # scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            raise NotImplementedError
            # masks = im_detect_mask_aug(model, im, boxes, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            raise NotImplementedError
            # heatmaps = im_detect_keypoints_aug(model, im, boxes, blob_conv)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes, blob_conv)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    return cls_boxes, cls_segms, cls_keyps


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
    inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]

    return_dict = model(**inputs)

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # cls prob (activations after softmax)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_mask(model, im_scale, boxes, blob_conv):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    pred_masks = model.module.mask_net(blob_conv, inputs)
    pred_masks = pred_masks.data.cpu().numpy().squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_keypoints(model, im_scale, boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
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
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

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
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
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
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale
