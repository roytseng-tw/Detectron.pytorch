from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import skimage.io as skio

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import datasets
from roi_data_layer.roidb import combined_roidb
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms_wrapper import nms
from model.utils.config import cfg, cfg_from_file, cfg_from_list
import model.utils.net_utils as net_utils
import model.utils.blob as blob_utils
import model.utils.misc as misc_utils
import model.utils.test as test_utils
import model.utils.vis as vis_utils

from model.mask_rcnn.resnet import resnet
import detectron_weights_loader as dwl


def parse_args():
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--detectron_arch',
                      help='Use architecture settings as in detectron',
                      default=True, type=distutils.util.strtobool)

  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

  parser.add_argument('--load_ckpt',
                      help='path of checkpoint to load')
  parser.add_argument('--load_detectron',
                      help='path to the detectron weight pickle file')

  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo', default="images")
  parser.add_argument('--output_dir',
                      help='directory to save demo results', default="mask_vis")
  parser.add_argument('--merge_pdfs', type=distutils.util.strtobool,
                      default=True)

  args = parser.parse_args()

  return args


def _get_image_blob(im):
  """Converts an image into a network input. Preprocessing: subtract mean and resize.
  Arguments:
    im (ndarray): a color image in **RGB** order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
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

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
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


if __name__ == '__main__':

  args = parse_args()
  print('Called with args:')
  print(args)

  if args.dataset == "coco":
    imdb = datasets.coco_mask.coco_mask('val', '2017')
  else:
    raise NotImplementedError

  args.cfg_file = "cfgs/{}_mask_ls.yml".format(args.net) if args.large_scale \
    else "cfgs/{}_mask.yml".format(args.net)

  if args.detectron_arch:
    args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]',
                     'RPN.CLS_ACTIVATION', 'sigmoid', 'RPN.OUT_DIM_AS_IN_DIM', 'True']
  else:
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  if args.load_detectron:
    cfg.POOLING_SIZE = 14
    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_POST_NMS_TOP_N = 1000
    cfg.TEST.MAX_SIZE = 1333
    cfg.TEST.SCALES = (800,)
    cfg.TEST.NMS = 0.5

  print('Using config:')
  pprint.pprint(cfg)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # Set random seed
  # np.random.seed(cfg.RNG_SEED)
  imdb.competition_mode(on=True)

  # initilize the network here.
  # pretrained = True if args.load_detectron else False  # True is for the bn weights
  if args.net == 'res50':
    maskRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    maskRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    maskRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    sys.exit("network is not defined")

  maskRCNN.create_architecture()

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
      dwl.load_detectron_weight(maskRCNN, args.load_detectron, dwl.mask_rcnn_R50_C4)
    else:
      raise NotImplementedError

    # mimic the Detectron affinechannel op
    def set_bn_stats(m):
      if isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.running_mean, 0)
        nn.init.constant(m.running_var, 1)
    maskRCNN.apply(set_bn_stats)

  if args.mGPUs:
    maskRCNN = nn.DataParallel(maskRCNN)
  if args.cuda:
    maskRCNN.cuda()

  maskRCNN.eval()

  imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
  num_images = len(imglist)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  for i in xrange(num_images):
    print('img', i)
    im = skio.imread(os.path.join(args.image_dir, imglist[i]))
    if im.ndim == 2:
      im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

    im_data, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_info = np.array([[im_data.shape[1], im_data.shape[2], im_scales[0]]], dtype=np.float32)

    im_data = Variable(torch.from_numpy(im_data).permute(0, 3, 1, 2), volatile=True)
    im_info = Variable(torch.from_numpy(im_info), volatile=True)
    gt_boxes = Variable(torch.zeros(1, 1, 5), volatile=True)
    num_boxes = Variable(torch.zeros(1), volatile=True)
    gt_masks = Variable(torch.zeros(1), volatile=True)

    if args.cuda:
      im_data = im_data.cuda()
      im_info = im_info.cuda()
      gt_boxes = gt_boxes.cuda()
      num_boxes = num_boxes.cuda()
      gt_masks = gt_masks.cuda()

    rois, rois_label, cls_prob, bbox_pred, mask_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      loss_mask \
      = maskRCNN(im_data, im_info, gt_boxes, num_boxes, gt_masks)

    cls_boxes = test_utils.im_test_all(args, im_info, rois, rois_label, cls_prob, bbox_pred, mask_pred)

    imname, _ = os.path.splitext(imglist[i])
    vis_utils.vis_one_image(
      im, imname, args.output_dir,
      cls_boxes, thresh=0.5, box_alpha=0.7, dataset=imdb, show_class=True)

  if args.merge_pdfs:
    merge_out_path = '{}/results.pdf'.format(args.output_dir)
    if os.path.exists(merge_out_path):
      os.remove(merge_out_path)
    command = "pdfunite {}/*.pdf {}".format(args.output_dir, merge_out_path)
    subprocess.call(command, shell=True)
