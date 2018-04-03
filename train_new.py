""" Training Script """

import argparse
import distutils.util
import os
import sys
import pickle
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets_new.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=100, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Epoch
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=6, type=int)

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers', help='number of worker to load data',
        default=10, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay learning rate. Epoch is 0-indexed. '
             'Decay on the start of epoch',
        default=[4, 5], nargs='+', type=int)

    # Resume training TODO: add resume training mechanism
    parser.add_argument(
        '--r', dest='resume',
        help='resume checkpoint or not',
        action='store_true')
    parser.add_argument('--checkrun', help='run name to load model')
    parser.add_argument('--checkepoch', help='epoch to load model', type=int)
    parser.add_argument('--checkstep', help='step to load model', type=int)

    # Checkpoint and Logging
    parser.add_argument(
        '--output_base_dir',
        help='Output base directory',
        default="Outputs")

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=3, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def save(output_dir, args, epoch, step, model, optimizer, iters_per_epoch):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'mask_rcnn_{}_{}.pth'.format(epoch, step))
    if args.mGPUs:
        model = model.module
    torch.save({
        'epoch': epoch,
        'step': step,
        'iters_per_epoch': iters_per_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE}, save_name)
    print('save model: {}'.format(save_name))


def main():
    """Main function"""
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('Batch size change from {} (in config file) to {}'.format(
        original_batch_size, args.batch_size))
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    ### Adjust learning based on batch size change linearly
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch size change: {} --> {}'.format(
        old_base_lr, cfg.SOLVER.BASE_LR))

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma

    if cfg.NUM_GPUS > 1:
        args.mGPUs = True

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index, im_sizes_list = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    sampler = MinibatchSampler(ratio_list, ratio_index, im_sizes_list)
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

    assert_and_infer_cfg()

    ### Model ###
    maskRCNN = Generalized_RCNN(train=True)

    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name)
        maskRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            assert cfg.POOLING_MODE == checkpoint['pooling_mode']

    if args.load_detectron:
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    if args.mGPUs:
        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True)

    if cfg.CUDA:
        maskRCNN.cuda()

    ### Optimizer ###
    bias_params = []
    nonbias_params = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': cfg.SOLVER.BASE_LR,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    lr = cfg.SOLVER.BASE_LR  # for display in command line

    ### Training Setups ###
    run_name = misc_utils.get_run_name()
    output_dir = misc_utils.get_output_dir(args, run_name)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    maskRCNN.train()

    iters_per_epoch = int(train_size / args.batch_size)  # drop last
    ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch
    step = 0
    try:
        logger.info('Training starts !')
        for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
            # ---- Start of epoch ----
            loss_avg = 0
            timers['train_loop'].tic()

            # adjust learning rate
            if args.lr_decay_epochs and epoch == args.lr_decay_epochs[0]:
                args.lr_decay_epochs.pop(0)
                net_utils.decay_learning_rate(optimizer, lr, cfg.SOLVER.GAMMA)
                lr *= cfg.SOLVER.GAMMA

            for step, input_data in zip(range(iters_per_epoch), dataloader):

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))

                outputs = maskRCNN(**input_data)

                rois_label = outputs['rois_label']
                cls_score = outputs['cls_score']
                bbox_pred = outputs['bbox_pred']
                loss_rpn_cls = outputs['loss_rpn_cls'].mean()
                loss_rpn_bbox = outputs['loss_rpn_bbox'].mean()
                loss_rcnn_cls = outputs['loss_rcnn_cls'].mean()
                loss_rcnn_bbox = outputs['loss_rcnn_bbox'].mean()

                loss = loss_rpn_cls + loss_rpn_bbox + loss_rcnn_cls + loss_rcnn_bbox

                if cfg.MODEL.MASK_ON:
                    loss_rcnn_mask = outputs['loss_rcnn_mask'].mean()
                    loss += loss_rcnn_mask

                if cfg.MODEL.KEYPOINTS_ON:
                    loss_rcnn_kps = outputs['loss_rcnn_kps'].mean()
                    loss += loss_rcnn_kps

                loss_avg += loss.data[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step+1) % ckpt_interval_per_epoch == 0:
                    save(output_dir, args, epoch, step, maskRCNN, optimizer, iters_per_epoch)

                if (step+1) % args.disp_interval == 0:
                    diff = timers['train_loop'].toc(average=False)
                    if step > 0:
                        loss_avg /= args.disp_interval

                    loss_rpn_cls = loss_rpn_cls.data[0]
                    loss_rpn_bbox = loss_rpn_bbox.data[0]
                    loss_rcnn_cls = loss_rcnn_cls.data[0]
                    loss_rcnn_bbox = loss_rcnn_bbox.data[0]
                    loss_rcnn_mask = loss_rcnn_mask.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                    print("[%s][epoch %2d][iter %4d / %4d]"
                          % (run_name, epoch, step, iters_per_epoch))
                    print("\t\tloss: %.4f, lr: %.2e" % (loss_avg, lr))
                    print("\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, diff))
                    print("\t\trpn_cls: %.4f, rpn_bbox: %.4f, rcnn_cls: %.4f,"
                          "rcnn_bbox %.4f, rcnn_mask %.4f"
                          % (loss_rpn_cls, loss_rpn_bbox, loss_rcnn_cls,
                             loss_rcnn_bbox, loss_rcnn_mask))
                    if args.use_tfboard:
                        info = {
                            'loss': loss_avg,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_bbox,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_bbox,
                            'loss_rcnn_mask': loss_rcnn_mask
                        }
                        for tag, value in info.items():
                            tblogger.add_scalar(tag, value, iters_per_epoch * epoch + step)

                    loss_avg = 0
                    timers['train_loop'].tic()

            # ---- End of epoch ----
            # save checkpoint
            save(output_dir, args, epoch, step, maskRCNN, optimizer, iters_per_epoch)
            # reset timer
            timers['train_loop'].reset()

    except (RuntimeError, KeyboardInterrupt) as e:
        print('Save on exception:', e)
        save(output_dir, args, epoch, step, maskRCNN, optimizer, iters_per_epoch)
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        # ---- Training ends ----
        if args.use_tfboard:
            tblogger.close()


if __name__ == '__main__':
    main()
