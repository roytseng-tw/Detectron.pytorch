# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import scipy.ndimage
from scipy.misc import imread
from pycocotools import mask as COCOmask
from core.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  # -- Force to exlude all annotations that are 'iscrowd' --
  # if cfg.TRAIN.USE_ALL_GT:
  #   # Include all ground truth boxes
  #   gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  # else:
  # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
  gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

  # Get masks for the single image, exclude ''iscrowd''. [1, num_classes, img_height, img_width]
  blobs['gt_masks'] = _get_seg_blob(roidb[0], im_scales[0], gt_inds)
  assert blobs['gt_masks'].shape[1] == len(gt_inds)

  blobs['gt_poses'] = roidb[0]['poses'][gt_inds, :, :].astype(np.float32)
  blobs['gt_poses'][:, :, :2] *= im_scales[0]

  # gt boxes: (x1, y1, x2, y2, cls)
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)  # [[h, w, scale]]

  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images [n, h, w, c]
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _get_seg_blob(roidb, im_scale, gt_inds):
  '''
  roidb: single roidb
  im_scale: scale for resize the mask
  gt_inds: indices of gt to use
  ---
  '''
  n_segs = len(roidb['segs'])
  masks = np.empty((1, n_segs, roidb['height'], roidb['width']), dtype=bool)
  for cnt, ind in enumerate(gt_inds):
    masks[0, cnt] = COCOmask.decode(roidb['segs'][ind])
  if roidb['flipped']:
    masks = masks[:, :, :, ::-1]
  masks = scipy.ndimage.zoom(masks, [1, 1, im_scale, im_scale], order=0)
  return masks
