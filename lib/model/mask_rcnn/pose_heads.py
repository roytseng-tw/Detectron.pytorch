import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.utils.config import cfg


# ---------------------------------------------------------------------------- #
# Pose R-CNN outputs and losses
# ---------------------------------------------------------------------------- #
class pose_outputs(nn.Module):
  def __init__(self, inplanes, n_kp_classes):
    super().__init__()
    self.classify = nn.ConvTranspose2d(inplanes, n_kp_classes, 4, 2, int(4/2-1))
    self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    init.normal(self.classify.weight, std=0.001)
    init.constant(self.classify.bias, 0)

  def forward(self, x):
    x = self.classify(x)
    x = self.upsample(x)
    return x

def pose_loss(pose_pred, rois_pose, weight):
  losses = F.cross_entropy(pose_pred.view(-1, cfg.KRCNN.HEATMAP_SIZE ** 2), rois_pose.long(), reduce=False)
  loss = torch.mean(losses * weight)
  return loss


# ---------------------------------------------------------------------------- #
# Pose heads
# ---------------------------------------------------------------------------- #
class pose_head_v1convX(nn.Module):
  def __init__(self, inplanes):
    super().__init__()
    hidden_dim = 512
    kernel_size = 3
    pad_size = kernel_size // 2
    module_list = []
    for _ in range(cfg.KRCNN.NUM_STACKED_CONVS):
      module_list.append(nn.Conv2d(inplanes, hidden_dim, kernel_size, 1, pad_size))
      module_list.append(nn.ReLU(inplace=True))
      inplanes = hidden_dim
    self.convX = nn.Sequential(*module_list)
    self.apply(self._init_weights)

    self.outplanes = hidden_dim

  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
      init.normal(m.weight, std=0.01)
      init.constant(m.bias, 0)

  def forward(self, x):
    x = self.convX(x)
    return x
