from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
from core.config import cfg, cfg_from_file, cfg_from_list
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
import datasets_new.dummy_datasets as datasets
import model.utils.net_utils as net_utils
import utils.blob as blob_utils
import utils.vis as vis_utils
import utils.misc as misc_utils


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='training dataset',
        default='coco')
    parser.add_argument(
        '--net',
        dest='net',
        help='res50, res101, res152',
        default='res50')

    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[],
        nargs=argparse.REMAINDER)

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='directory to load images for demo',
        default="images")
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="mask_vis")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


def _get_image_blob(im):
    """Converts an image into a network input. Preprocessing: subtract mean and resize.
    Arguments:
        im (ndarray): a color image in **RGB** order
    Returns:
        blob (ndarray): a data blob holding an image pyramid.
                        Axis order: (batch elem, channel, height, width)
        im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig = im_orig[:, :, ::-1]  # RGB to BGR
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    im_scale = float(cfg.TEST.SCALE) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def main():
    """main function"""
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    if args.cfg_file is None:
        args.cfg_file = "cfgs/{}_mask.yml".format(args.net)
    print('cfg file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    maskRCNN = Generalized_RCNN(train=False)

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        maskRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            assert cfg.POOLING_MODE == checkpoint['pooling_mode']

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        if args.net == 'res50':
            load_detectron_weight(maskRCNN, args.load_detectron)
        else:
            raise NotImplementedError

        # mimic the Detectron affinechannel op
        def set_bn_stats(m):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.running_mean, 0)
                nn.init.constant(m.running_var, 1)

        maskRCNN.apply(set_bn_stats)

    if args.cuda:
        maskRCNN.cuda()

    maskRCNN.eval()

    imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in xrange(num_images):
        print('img', i)
        im = cv2.imread(os.path.join(args.image_dir, imglist[i]))

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)

        im_name, _ = os.path.splitext(imglist[i])
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

    if args.merge_pdfs:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
