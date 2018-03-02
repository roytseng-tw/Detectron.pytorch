from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import skimage.transform as sktf
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes, gt_masks, gt_poses):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois CHECK: SEEMS an important trick!
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights, masks, masks_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes, gt_masks, gt_poses)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, masks, masks_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, gt_masks, gt_poses):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1).index(offset.view(-1))\
                                                            .view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()

        gt_masks = gt_masks.cpu()  # convert to cpu, because of resizing later
        # move following tensor to gpu later
        gt_masks_batch = gt_masks.new(batch_size, rois_per_image, cfg.TRAIN.MASK_SHAPE[0], cfg.TRAIN.MASK_SHAPE[1]).zero_()
        masks_weights = gt_masks.new(batch_size, rois_per_image).zero_()

        gt_poses_keep = gt_poses.new(batch_size, rois_per_image, gt_poses.size(2), gt_poses.size(3)).zero_()
        # [_, _, n_kps]: contains the 1D-location (y * HEATMAP_SIZE + x) for each kp.  gt_pose is a float tensor
        gt_poses_batch = torch.IntTensor(batch_size, rois_per_image, gt_poses.size(2)).zero_()
        # poses_weights: clone from masks_weights later

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
            gt_poses_keep[i] = gt_poses[i][gt_assignment[i][keep_inds]]

            # cropped to bbox boundaries and resized to neural network output size
            # use masks_weights to select valid masks: foreground and rounded box area > 0
            if fg_num_rois > 0:
                assign_inds = gt_assignment[i][fg_inds].cpu()
                masks = gt_masks[i][assign_inds]
                masks_weights[i, :fg_num_rois] = 1
                # bboxes = all_rois[i][fg_inds].cpu().type(torch.IntTensor)
                bboxes = all_rois[i][fg_inds].cpu().round().type(torch.IntTensor)
                for cnt, (mask, bbox) in enumerate(zip(masks, bboxes)):
                    _, x1, y1, x2, y2 = bbox
                    if x1 == x2 or y1 == y2:  # rounded box area == 0, mask cannot be resize
                        masks_weights[i, cnt] = 0
                    else:
                        # -- Mask --
                        mask_re = sktf.resize(mask[y1:y2, x1:x2].numpy(), cfg.TRAIN.MASK_SHAPE, order=0).astype(np.float32)
                        # print(mask_re.dtype, set(mask_re.reshape(-1).tolist())) --> {0.0, 1.0}
                        gt_masks_batch[i][cnt].copy_(torch.from_numpy(mask_re))
                        # -- Pose --
                        scale_h = cfg.KRCNN.HEATMAP_SIZE / (y2 - y1)
                        scale_w = cfg.KRCNN.HEATMAP_SIZE / (x2 - x1)
                        gt_poses_keep[i][cnt, :, :2] *= gt_poses.new([[[scale_h, scale_w, 1]]])

        # round pose to integer position
        gt_poses_keep = gt_poses_keep.round().type(torch.IntTensor)
        gt_poses_batch = gt_poses_keep[:, :, 1] * cfg.KRCNN.HEATMAP_SIZE + gt_poses_keep[:, :, 2]

        # Calculate pose weights
        poses_weights = masks_weights.clone()
        vis = gt_poses_keep[:, :, :, 2]
        nonvis_ind = torch.nonzero(vis.view(-1) == 0)  # equal is safe because gt_poses_keep has been round to int
        poses_weights = poses_weights.view(-1)
        poses_weights[nonvis_ind] = 0
        poses_weights = poses_weights.view_as(masks_weights)

        # move to corresponding gpu
        gt_masks_batch = gt_masks_batch.cuda(labels_batch.get_device())
        masks_weights = masks_weights.cuda(labels_batch.get_device())

        bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)  # [batch_size, rois_per_image, 4]

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights, gt_masks_batch, masks_weights, gt_poses_batch, poses_weights
