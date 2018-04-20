""" Training script for steps_with_decay policy"""

import argparse
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
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer

 # Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)


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
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

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

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    # Checkpoint and Logging
    parser.add_argument(
        '--output_base_dir',
        help='Output base directory',
        default="Outputs")

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if args.mGPUs:
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('Batch size change from {} (in config file) to {}'.format(
        original_batch_size, args.batch_size))
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

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
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size

    sampler = MinibatchSampler(ratio_list, ratio_index)
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)

    ### Model ###
    maskRCNN = Generalized_RCNN()

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
    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
        if args.resume:
            args.start_step = checkpoint['step'] + 1
            assert checkpoint['train_size'] == train_size
            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

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

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] > args.start_step:
            decay_steps_ind = i
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    logger.info('Training starts !')
    loss_avg = 0
    try:
        timers['train_loop'].tic()
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                lr = cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = lr_new
                decay_steps_ind += 1

            try:
                input_data = next(dataiterator)
            except StopIteration:
                dataiterator = iter(dataloader)
                input_data = next(dataiterator)

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
                loss_rcnn_keypoints = outputs['loss_rcnn_keypoints'].mean()
                loss += loss_rcnn_keypoints

            loss_avg += loss.data.cpu().numpy()[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

            if ((step % args.disp_interval == 0 and
                 (step - args.start_step >= args.disp_interval)) or
                    step == cfg.SOLVER.MAX_ITER - 1):
                diff = timers['train_loop'].toc(average=False)
                loss_avg /= args.disp_interval

                loss_rpn_cls = loss_rpn_cls.data[0]
                loss_rpn_bbox = loss_rpn_bbox.data[0]
                loss_rcnn_cls = loss_rcnn_cls.data[0]
                loss_rcnn_bbox = loss_rcnn_bbox.data[0]
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
                print("[ %s ][ step %d ]" % (run_name, step))
                print("\t\tloss: %.4f, lr: %.2e" % (loss_avg, lr))
                print("\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, diff))
                print("\t\trpn_cls: %.4f, rpn_bbox: %.4f, rcnn_cls: %.4f, rcnn_bbox %.4f"
                    % (loss_rpn_cls, loss_rpn_bbox, loss_rcnn_cls, loss_rcnn_bbox))

                print_prefix = "\t\t"
                if cfg.MODEL.MASK_ON:
                    loss_rcnn_mask = loss_rcnn_mask.data[0]
                    print("%srcnn_mask %.4f" % (print_prefix, loss_rcnn_mask))
                    print_prefix = ", "
                if cfg.MODEL.KEYPOINTS_ON:
                    loss_rcnn_keypoints = loss_rcnn_keypoints.data[0]
                    print("%srcnn_keypoints %.4f" % (print_prefix, loss_rcnn_keypoints))

                if args.use_tfboard and not args.no_save:
                    info = {
                        'lr': lr,
                        'loss': loss_avg,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_bbox,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_bbox,
                    }
                    if cfg.MODEL.MASK_ON:
                        info['loss_rcnn_mask'] = loss_rcnn_mask
                    if cfg.MODEL.KEYPOINTS_ON:
                        info['loss_rcnn_keypoints'] = loss_rcnn_keypoints
                    for tag, value in info.items():
                        tblogger.add_scalar(tag, value, step)

                loss_avg = 0
                timers['train_loop'].tic()

    except (RuntimeError, KeyboardInterrupt) as e:
        print('Save on exception:', e)
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        # ---- Training ends ----
        if args.use_tfboard and not args.no_save:
            tblogger.close()


if __name__ == '__main__':
    main()
