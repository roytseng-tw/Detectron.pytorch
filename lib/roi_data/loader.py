import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as sampler
from torch.utils.data.dataloader import default_collate

from core.config import cfg
from roi_data.minibatch import get_minibatch
import utils.blob as blob_utils
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes


class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)

    def __getitem__(self, index_tuple):
        index, ratio, imsizes = index_tuple
        single_db = [self._roidb[index]]
        blobs = get_minibatch(single_db)

        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        if self._roidb[index]['need_crop']:
            self.crop_data(blobs, ratio)
            # Check bounding box
            entry = blobs['roidb'][0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]

        # Padding the image based on the ratio
        self.pad_data(blobs, imsizes)

        blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we should serialize in collate_fn

        # for key, v in blobs.items():
        #     if key != 'roidb':
        #         print(key, v.shape)

        return blobs

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: the mechnism for the case box_region > size_crop rethinking
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :,]
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:,0]))
            max_x = math.floor(np.max(boxes[:,2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes

    def pad_data(self, blobs, imsizes):
        data_height, data_width = map(int, blobs['im_info'][:2])
        target_height, target_width = map(int, imsizes[0])
        data_padded = np.zeros((3, target_height, target_width), dtype=np.float32)
        data_padded[:, :data_height, :data_width] = blobs['data']
        blobs['data'] = data_padded
        blobs['im_info'][:2] = data_padded.shape[1:]

    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


def cal_minibatch_imsizes(im_sizes_list):
    """Given the ratio_list, we want to make the SIZE same for each minibatch on each GPU.
    This should work for considering cfg.TRAIN.MAX_SIZE and multiple cfg.TRAIN.SCALES (#TODO maybe).
    """
    DATA_SIZE = len(im_sizes_list)
    imsizes_list_minibatch = np.empty((DATA_SIZE, len(cfg.TRAIN.SCALES), 2))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        imsizes = im_sizes_list[left_idx:(right_idx+1)]

        # following logic is designed for single cfg.TRAIN.SCALES
        # for multi scales we should probably grouping images with similar scales as well as ratio
        imsizes_list_minibatch[left_idx:(right_idx+1)] = np.max(imsizes.reshape(-1, 2), axis=0)

    return imsizes_list_minibatch


class MinibatchSampler(sampler.Sampler):
    def __init__(self, ratio_list, ratio_index, im_sizes_list):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.im_sizes_list = im_sizes_list
        self.num_data = len(ratio_list)

        if cfg.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)
            self.imsizes_list_minibatch = cal_minibatch_imsizes(im_sizes_list)

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, cfg.TRAIN.IMS_PER_BATCH)
            round_num_data = n * cfg.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            npr.shuffle(indices.reshape(-1, cfg.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
            imsizes_list_minibatch = self.imsizes_list_minibatch[indices]
        else:
            rand_perm = npr.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            im_sizes_list = self.im_sizes_list[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)
            imsizes_list_minibatch = cal_minibatch_imsizes(im_sizes_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist(),
                        imsizes_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch
