import argparse
import distutils.util
import os
import sys
import pickle
import pprint
import traceback
import logging
from collections import defaultdict

import numpy as np
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import torch
from torch.autograd import Variable
import torch.nn as nn

import _init_paths
from datasets_new.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch
from core.config import cfg, cfg_from_file, cfg_from_list
from utils.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='training dataset',
        default='pascal_voc',
        type=str)
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16, res101, res50 ...',
        default='vgg16',
        type=str)
    parser.add_argument(
        '--ls',
        dest='large_scale',
        help='whether use large imag scale',
        action='store_true')
    parser.add_argument(
        '--cag',
        dest='class_agnostic',
        help='whether perform class_agnostic bbox regression',
        action='store_true')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=[],
        nargs=argparse.REMAINDER)

    parser.add_argument(
        '--detectron_arch',
        help='Use architecture settings as in detectron',
        default=True,
        type=distutils.util.strtobool)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='directory to save models',
        default=os.path.join(os.environ['HOME'], "models"))
    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save for each epoch. '
        'Not include the one at the end of an epoch.',
        default=3,
        type=int)

    parser.add_argument(
        '--start_epoch',
        dest='start_epoch',
        help='starting epoch',
        default=0,
        type=int)
    parser.add_argument(
        '--epochs',
        dest='num_epochs',
        help='number of epochs to train',
        default=10,
        type=int)
    parser.add_argument(
        '--disp_interval',
        dest='disp_interval',
        help='number of iterations to display',
        default=100,
        type=int)

    parser.add_argument(
        '--nw',
        dest='num_workers',
        help='number of worker to load data',
        default=0,
        type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')
    parser.add_argument(
        '--bs', dest='batch_size', help='batch_size', default=1, type=int)

    # config optimization
    parser.add_argument(
        '--o',
        dest='optimizer',
        help='training optimizer',
        default="sgd",
        type=str)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='starting learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--lr_decay_step',
        dest='lr_decay_step',
        help='step to do learning rate decay, unit is epoch',
        default=5,
        type=int)
    parser.add_argument(
        '--lr_decay_gamma',
        dest='lr_decay_gamma',
        help='learning rate decay ratio',
        default=0.1,
        type=float)

    # set training session
    parser.add_argument(
        '--s',
        dest='session',
        help='training session, for recognization only. '
        'Shown in terminal outputs.',
        default=1,
        type=int)

    # resume trained model
    parser.add_argument(
        '--r',
        dest='resume',
        help='resume checkpoint or not',
        action='store_true')
    parser.add_argument('--checkrun', help='run name to load model')
    parser.add_argument('--checkepoch', help='epoch to load model', type=int)
    parser.add_argument('--checkstep', help='step to load model', type=int)

    # load checkpoint
    parser.add_argument('--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    # log and diaplay
    parser.add_argument(
        '--use_tfboard',
        dest='use_tfboard',
        help='whether use tensorflow tensorboard',
        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, 'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    if args.dataset == "coco2017":
        args.train_datasets = ('coco_2017_train', )
        args.train_proposal_files = ()
    else:
        sys.exit('Unexpect args.dataset value: ', args.dataset)
    # cfg.TRAIN.BATCH_SIZE_PRE_IM = 120  # chance to OOM if 128 on 1080ti

    if args.net == 'res50':
        args.cfg_file = "cfgs/res50_mask.yml"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # print('Using config:')
    # pprint.pprint(cfg)

    timers = defaultdict(Timer)

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        args.train_datasets, args.train_proposal_files)
    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb' % timers['roidb'].average_time)


    # FIXME: manually set for now
    cfg.MODEL.NUM_CLASSES = 81

    sampler = MinibatchSampler(ratio_list, ratio_index)
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_minibatch)
    data_iter = iter(dataloader)
    data = next(data_iter)

    from IPython import embed
    embed()
