# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import numpy as np
import argparse
import pprint
import pdb
import time
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader_pose_mask import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.misc import get_run_name
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
  adjust_learning_rate, save_checkpoint, clip_gradient

# from model.mask_rcnn.vgg16 import vgg16
from model.mask_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='vgg16', type=str)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large image scale',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default=os.path.join(os.environ['HOME'], "models"))
  parser.add_argument('--ckpt_num_per_epoch',
                      help='number of checkpoints to save for each epoch. '
                           'Not include the one at the end of an epoch.',
                      default=3, type=int)

  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=0, type=int)
  parser.add_argument('--epochs', dest='num_epochs',
                      help='number of epochs to train',
                      default=10, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)

  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session, for recognization only. '
                           'Shown in terminal outputs.',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      action='store_true')
  parser.add_argument('--checkrun',
                      help='run name to load model')
  parser.add_argument('--checkepoch',
                      help='epoch to load model',
                      type=int)
  parser.add_argument('--checkstep',
                      help='step to load model',
                      type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0, batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def save(args, epoch, step, model, optimizer, iters_per_epoch):
  ckpt_dir = os.path.join(output_dir, 'ckpt')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  save_name = os.path.join(ckpt_dir, 'pose_mask_rcnn_{}_{}.pth'.format(epoch, step))
  if args.mGPUs:
    model = model.module
  save_checkpoint({
    'epoch': epoch,
    'step': step,
    'iters_per_epoch': iters_per_epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'pooling_mode': cfg.POOLING_MODE,
    'class_agnostic': args.class_agnostic,
  }, save_name)
  print('save model: {}'.format(save_name))


if __name__ == '__main__':

  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "coco2017":  # for mask rcnn
      args.imdb_name = "coco-mask-pose_2017_train"
      args.imdbval_name = "coco-mask-pose_2017_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50', 'KRCNN.NUM_KEYPOINTS', '17']
  elif args.dataset == "coco2014":  # for mask rcnn
      args.imdb_name = "coco-mask-pose_2014_train"
      args.imdbval_name = "coco-mask-pose_2014_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50', 'KRCNN.NUM_KEYPOINTS', '17']

  args.cfg_file = "cfgs/{}_mask_ls.yml".format(args.net) if args.large_scale else "cfgs/{}_mask.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.HAS_POSE_BRANCH = True
  cfg.POOLING_SIZE = 14
  cfg.TRAIN.MASK_SHAPE = [28, 28]
  cfg.KRCNN.HEATMAP_SIZE = 56  # ROI_XFORM_RESOLUTION (14) * UP_SCALE (2) * USE_DECONV_OUTPUT (2)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  run_name = get_run_name()
  output_dir = os.path.join(args.save_dir, args.net, args.dataset, run_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    # Set the logger
    logger = SummaryWriter(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  gt_masks = torch.FloatTensor(1)
  gt_poses = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    # gt_masks = gt_masks.cuda(), move rois_mask to cuda later is enough
    # gt_poses = gt_poses.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  gt_masks = Variable(gt_masks)
  gt_poses = Variable(gt_poses)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    maskRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    maskRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    maskRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    maskRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  maskRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  # tr_momentum = cfg.TRAIN.MOMENTUM
  # tr_momentum = args.momentum

  params = []
  for key, value in dict(maskRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params': [value], 'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                   'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params': [value], 'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)
  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'pose_mask_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch'] + 1  # Assume to start from next epoch
    maskRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    maskRCNN = nn.DataParallel(maskRCNN)

  if args.cuda:
    maskRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)  # drop last
  ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch

  for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
    # setting to train mode
    maskRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % args.lr_decay_step == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)

      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      gt_masks.data.resize_(data[4].size()).copy_(data[4])
      gt_poses.data.resize_(data[5].size()).copy_(data[5])

      rois, rois_label, cls_prob, bbox_pred, mask_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        loss_mask, loss_pose \
        = maskRCNN(im_data, im_info, gt_boxes, num_boxes, gt_masks, gt_poses)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
        + loss_mask.mean() + loss_pose.mean()
      loss_temp += loss.data[0]

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
        clip_gradient(maskRCNN, 10.)
      optimizer.step()

      if step != 0 and step % ckpt_interval_per_epoch == 0:
        save(args, epoch, step, maskRCNN, optimizer, iters_per_epoch)

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        # .mean() in case of mGPUs
        loss_rpn_cls = rpn_loss_cls.mean().data[0]
        loss_rpn_box = rpn_loss_box.mean().data[0]
        loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
        loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
        loss_mask = loss_mask.mean().data[0]
        loss_pose = loss_pose.mean().data[0]
        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt

        print("[%s][session %d][epoch %2d][iter %4d / %4d]"
              % (run_name, args.session, epoch, step, iters_per_epoch))
        print("\t\tloss: %.4f, lr: %.2e" % (loss_temp, lr))
        print("\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, mask %.4f, pose %.4f"
              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_mask, loss_pose))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'loss_mask': loss_mask,
            'loss_pose': loss_pose
          }
          for tag, value in info.items():
            logger.add_scalar(tag, value, iters_per_epoch * epoch + step)

        loss_temp = 0
        start = time.time()

    # save at the end of each epoch
    save(args, epoch, step, maskRCNN, optimizer, iters_per_epoch)
    end = time.time()
    print(end - start)

  # Training ends
  logger.close()
