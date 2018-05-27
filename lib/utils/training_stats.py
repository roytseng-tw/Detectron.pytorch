#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import datetime
import numpy as np

from core.config import cfg
from utils.logging import log_stats
from utils.logging import SmoothedValue
from utils.timer import Timer
import utils.net as nu


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, misc_args, log_period=20, tensorboard_logger=None):
        # Output logging period in SGD iterations
        self.misc_args = misc_args
        self.LOG_PERIOD = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['iter', 'eta']
        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 20
        def create_smoothed_value():
            return SmoothedValue(self.WIN_SZ)
        self.smoothed_losses = defaultdict(create_smoothed_value)
        self.smoothed_metrics = defaultdict(create_smoothed_value)
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        # For the support of args.iter_size
        self.inner_total_loss = []
        self.inner_losses = defaultdict(list)
        if cfg.FPN.FPN_ON:
            self.inner_loss_rpn_cls = []
            self.inner_loss_rpn_bbox = []
        self.inner_metrics = defaultdict(list)

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, model_out, inner_iter=None):
        """Update tracked iteration statistics."""
        if inner_iter is not None and self.misc_args.iter_size > 1:
            # For the case of using args.iter_size > 1
            return self._UpdateIterStats_inner(model_out, inner_iter)

        # Following code is saved for compatability of train_net.py and iter_size==1
        total_loss = 0
        if cfg.FPN.FPN_ON:
            loss_rpn_cls_data = 0
            loss_rpn_bbox_data = 0

        for k, loss in model_out['losses'].items():
            assert loss.shape[0] == cfg.NUM_GPUS
            loss = loss.mean(dim=0, keepdim=True)
            total_loss += loss
            loss_data = loss.data[0]
            model_out['losses'][k] = loss
            if cfg.FPN.FPN_ON:
                if k.startswith('loss_rpn_cls_'):
                    loss_rpn_cls_data += loss_data
                elif k.startswith('loss_rpn_bbox_'):
                    loss_rpn_bbox_data += loss_data
            self.smoothed_losses[k].AddValue(loss_data)

        model_out['total_loss'] = total_loss  # Add the total loss for back propagation
        self.smoothed_total_loss.AddValue(total_loss.data[0])
        if cfg.FPN.FPN_ON:
            self.smoothed_losses['loss_rpn_cls'].AddValue(loss_rpn_cls_data)
            self.smoothed_losses['loss_rpn_bbox'].AddValue(loss_rpn_bbox_data)

        for k, metric in model_out['metrics'].items():
            metric = metric.mean(dim=0, keepdim=True)
            self.smoothed_metrics[k].AddValue(metric.data[0])

    def _UpdateIterStats_inner(self, model_out, inner_iter):
        """Update tracked iteration statistics for the case of iter_size > 1"""
        assert inner_iter < self.misc_args.iter_size

        total_loss = 0
        if cfg.FPN.FPN_ON:
            loss_rpn_cls_data = 0
            loss_rpn_bbox_data = 0

        if inner_iter == 0:
            self.inner_total_loss = []
            for k in model_out['losses']:
                self.inner_losses[k] = []
            if cfg.FPN.FPN_ON:
                self.inner_loss_rpn_cls = []
                self.inner_loss_rpn_bbox = []
            for k in model_out['metrics']:
                self.inner_metrics[k] = []

        for k, loss in model_out['losses'].items():
            assert loss.shape[0] == cfg.NUM_GPUS
            loss = loss.mean(dim=0, keepdim=True)
            total_loss += loss
            loss_data = loss.data[0]

            model_out['losses'][k] = loss
            if cfg.FPN.FPN_ON:
                if k.startswith('loss_rpn_cls_'):
                    loss_rpn_cls_data += loss_data
                elif k.startswith('loss_rpn_bbox_'):
                    loss_rpn_bbox_data += loss_data

            self.inner_losses[k].append(loss_data)
            if inner_iter == (self.misc_args.iter_size - 1):
                loss_data = self._mean_and_reset_inner_list('inner_losses', k)
                self.smoothed_losses[k].AddValue(loss_data)

        model_out['total_loss'] = total_loss  # Add the total loss for back propagation
        total_loss_data = total_loss.data[0]
        self.inner_total_loss.append(total_loss_data)
        if cfg.FPN.FPN_ON:
            self.inner_loss_rpn_cls.append(loss_rpn_cls_data)
            self.inner_loss_rpn_bbox.append(loss_rpn_bbox_data)
        if inner_iter == (self.misc_args.iter_size - 1):
            total_loss_data = self._mean_and_reset_inner_list('inner_total_loss')
            self.smoothed_total_loss.AddValue(total_loss_data)
            if cfg.FPN.FPN_ON:
                loss_rpn_cls_data = self._mean_and_reset_inner_list('inner_loss_rpn_cls')
                loss_rpn_bbox_data = self._mean_and_reset_inner_list('inner_loss_rpn_bbox')
                self.smoothed_losses['loss_rpn_cls'].AddValue(loss_rpn_cls_data)
                self.smoothed_losses['loss_rpn_bbox'].AddValue(loss_rpn_bbox_data)

        for k, metric in model_out['metrics'].items():
            metric = metric.mean(dim=0, keepdim=True)
            metric_data = metric.data[0]
            self.inner_metrics[k].append(metric_data)
            if inner_iter == (self.misc_args.iter_size - 1):
                metric_data = self._mean_and_reset_inner_list('inner_metrics', k)
                self.smoothed_metrics[k].AddValue(metric_data)

    def _mean_and_reset_inner_list(self, attr_name, key=None):
        """Take the mean and reset list empty"""
        if key:
            mean_val = sum(getattr(self, attr_name)[key]) / self.misc_args.iter_size
            getattr(self, attr_name)[key] = []
        else:
            mean_val = sum(getattr(self, attr_name)) / self.misc_args.iter_size
            setattr(self, attr_name, [])
        return mean_val

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0 or
                cur_iter == cfg.SOLVER.MAX_ITER - 1):
            stats = self.GetStats(cur_iter, lr)
            log_stats(stats, self.misc_args)
            if self.tblogger:
                self.tb_log_stats(stats, cur_iter)

    def tb_log_stats(self, stats, cur_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self.tb_log_stats(v, cur_iter)
                else:
                    self.tblogger.add_scalar(k, v, cur_iter)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            cfg.SOLVER.MAX_ITER - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            iter=cur_iter + 1,  # 1-indexed
            time=self.iter_timer.average_time,
            eta=eta,
            loss=self.smoothed_total_loss.GetMedianValue(),
            lr=lr,
        )
        stats['metrics'] = OrderedDict()
        for k in sorted(self.smoothed_metrics):
            stats['metrics'][k] = self.smoothed_metrics[k].GetMedianValue()

        head_losses = []
        rpn_losses = []
        rpn_fpn_cls_losses = []
        rpn_fpn_bbox_losses = []
        for k, v in self.smoothed_losses.items():
            toks = k.split('_')
            if len(toks) == 2:
                head_losses.append((k, v.GetMedianValue()))
            elif len(toks) == 3:
                rpn_losses.append((k, v.GetMedianValue()))
            elif len(toks) == 4 and toks[2] == 'cls':
                rpn_fpn_cls_losses.append((k, v.GetMedianValue()))
            elif len(toks) == 4 and toks[2] == 'bbox':
                rpn_fpn_bbox_losses.append((k, v.GetMedianValue()))
            else:
                raise ValueError("Unexpected loss key: %s" % k)
        stats['head_losses'] = OrderedDict(head_losses)
        stats['rpn_losses'] = OrderedDict(rpn_losses)
        stats['rpn_fpn_cls_losses'] = OrderedDict(rpn_fpn_cls_losses)
        stats['rpn_fpn_bbox_losses'] = OrderedDict(rpn_fpn_bbox_losses)

        return stats
