import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from model.faster_rcnn import resnet


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #
class mask_outputs(nn.Module):
    def __init__(self, inplanes, n_classes):
        super().__init__()
        self.classify = nn.Conv2d(inplanes, n_classes, 1, 1, 0)
        init.kaiming_normal(self.classify.weight)
        init.constant(self.classify.bias, 0)

    def forward(self, x):
        x = self.classify(x)
        if not self.training:
            x = F.sigmoid(x)
        return x

def mask_losses(mask_pred, rois_mask, rois_label, weight):
    mask_gt = Variable(mask_pred.data.new(*mask_pred.size()).zero_())
    for n, l in enumerate(rois_label.data):
        mask_gt[n, l] = rois_mask[n]
    loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt, weight=weight.contiguous().view(-1, 1, 1, 1))
    return loss


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #
def _make_layer(block, inplanes, planes, n_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion), )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, n_blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class mask_head_v0up(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        # self.dilation = dilation

        self.res5 = _make_layer(resnet.Bottleneck, inplanes, 512, 3, stride=1)
        dim_reduced = 256
        self.upconv5 = nn.ConvTranspose2d(512 * 4, dim_reduced, 2, 2, 0)
        init.normal(self.upconv5.weight, std=0.001)
        init.constant(self.upconv5.bias, 0)

        self.outplanes = dim_reduced

    def forward(self, x):
        x = self.res5(x)
        # print(x.size()) e.g. (128, 2048, 7, 7)
        x = self.upconv5(x)
        x = F.relu(x, inplace=True)
        return x
