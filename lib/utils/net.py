import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg

logger = logging.getLogger(__name__)


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def decay_learning_rate(optimizer, cur_lr, decay_rate):
    """Decay learning rate"""
    new_lr = cur_lr * decay_rate
    # ratio = _get_lr_change_ratio(cur_lr, new_lr)
    ratio = 1 / decay_rate
    if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
        logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
    # Update learning rate, note that different parameter may have different learning rate
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        new_lr = decay_rate * param_group['lr']
        param_group['lr'] = new_lr
        if cfg.SOLVER.TYPE in ['SGD']:
            if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                    ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
                _CorrectMomentum(optimizer, param_group['params'], new_lr / cur_lr)


def _CorrectMomentum(optimizer, param_keys, correction):
    """The MomentumSGDUpdate op implements the update V as

        V := mu * V + lr * grad,

    where mu is the momentum factor, lr is the learning rate, and grad is
    the stochastic gradient. Since V is not defined independently of the
    learning rate (as it should ideally be), when the learning rate is
    changed we should scale the update history V in order to make it
    compatible in scale with lr * grad.
    """
    logger.info('Scaling update history by %.6f (new lr / old lr)', correction)
    for p_key in param_keys:
        optimizer.state[p_key]['momentum_buffer'] *= correction


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio


def affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid