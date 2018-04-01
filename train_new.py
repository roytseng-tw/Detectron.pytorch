""" Training Script """

import argparse
import distutils.util
import os
import sys
import traceback
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets_new.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from model.utils.net_utils import clip_gradient, adjust_learning_rate
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from utils.misc import get_run_name

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

    parser.add_argument(
        '--no_save',
        help='do not save anything',
        action='store_true')

    args = parser.parse_args()
    return args


def save(output_dir, args, epoch, step, model, optimizer, iters_per_epoch):
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
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))


if __name__ == '__main__':
    # pylint: disable=C0103

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, 'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if cfg.NUM_GPUS > 1:
        args.mGPUs = True

    if args.dataset == "coco2017":
        args.train_datasets = ('coco_2017_train', )
        args.train_proposal_files = ()
    else:
        sys.exit('Unexpect args.dataset value: ', args.dataset)
    # cfg.TRAIN.BATCH_SIZE_PRE_IM = 120  # chance to OOM if 128 on 1080ti

    if args.net == 'res50':
        args.cfg_file = "cfgs/res50_mask.yml"
    else:
        raise ValueError("No config file yet.")

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    assert len(cfg.TRAIN.SCALES) == 1, "Currently, only support single scale for data loading"

    timers = defaultdict(Timer)


    ### Dataset ###

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    timers['roidb'].tic()
    roidb, ratio_list, ratio_index, im_sizes_list = combined_roidb_for_training(
        args.train_datasets, args.train_proposal_files)
    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb' % timers['roidb'].average_time)


    # FIXME: manually set for now
    cfg.MODEL.NUM_CLASSES = 81

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
    data_iter = iter(dataloader)


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

        # mimic the Detectron affinechannel op
        def set_bn_stats(m):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.running_mean, 0)
                nn.init.constant(m.running_var, 1)
        maskRCNN.apply(set_bn_stats)

    if args.mGPUs:
        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True)

    if cfg.CUDA:
        maskRCNN.cuda()


    ### Optimizer ###

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    ### Training Setups ###

    run_name = get_run_name()
    output_dir = os.path.join(args.save_dir, args.net, args.dataset, run_name)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)


    ### Training Loop ###

    iters_per_epoch = int(train_size / args.batch_size)  # drop last
    ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch
    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
            maskRCNN.train()
            loss_avg = 0
            timers['train_loop'].tic()

            for step, input_data in zip(range(iters_per_epoch), dataloader):

                for key in input_data:
                    if key != 'roidb':
                        # roidb is list of ndarray not tensor,
                        # because roidb consists of entries of variable length
                        input_data[key] = list(map(Variable, input_data[key]))
                        # if cfg.CUDA:
                        #     input_data[key] = list(map(lambda x: x.cuda(), input_data[key]))

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
                if args.net == "vgg16":
                    clip_gradient(maskRCNN, 10)
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

                    print("[%s][session %d][epoch %2d][iter %4d / %4d]"
                          % (run_name, args.session, epoch, step, iters_per_epoch))
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
            # adjust learning rate
            if (epoch+1) % args.lr_decay_step == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma
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
