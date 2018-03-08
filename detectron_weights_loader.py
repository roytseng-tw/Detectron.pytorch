import _init_paths
import pickle
import torch

from model.mask_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_list


# ---------------------------------------------------------------------------- #
# Notes:
#  - Even there is no bias, detectron still has a tensor with zero values
#    (most of time) of name '*_b'
#  - There is NO **running_mean** and **running_var** in detectron weights
#  - detectron weights of name '*_momentum' can be ignore
#  - 'fc1000_w' and 'fc1000_b' in detectron weights can be ignored.
# ---------------------------------------------------------------------------- #

def mask_rcnn_R50_C4():
  # mapping names from detectron weights to torch weights
  name_mapping = {
    'conv1_w': 'RCNN_base.0.weigth',
    # 'conv1_b' : There should be no bias. However,
    #             small values around e-9,10 in detectron weight.
    'res_conv1_bn_s': 'RCNN_base.1.weight',
    'res_conv1_bn_b': 'RCNN_base.1.bias',
  }

  nblocks = [3, 4, 6, 3]  # res2, res3, res4, res5
  for block_id, n in enumerate(nblocks, start=2):
    for i in range(n):
      d_prefix = 'res%d_%d' % (block_id, i)  # name prefix of detectron weight
      if block_id < 3:
        p_prefix = 'RCNN_base.%d.%d' % (block_id + 2, i)  # name prefix of pytorch weight
      else: # res5
        p_prefix = 'RCNN_top.0.%d' % (i)

      # Shorcut connection
      name_mapping[d_prefix + '_branch1_w'] = p_prefix + '.downsample.0.weight'
      # resN_%d_branch1_b: no bias, all zeros. Omitting following comments.
      name_mapping[d_prefix + '_branch1_bn_s'] = p_prefix + '.downsample.1.weight'
      name_mapping[d_prefix + '_branch1_bn_b'] = p_prefix + '.downsample.1.bias'

      # Bottleneck block
      name_mapping[d_prefix + '_branch2a_w'] = p_prefix + '.conv1.weight'
      name_mapping[d_prefix + '_branch2a_bn_s'] = p_prefix + '.bn1.weight'
      name_mapping[d_prefix + '_branch2a_bn_b'] = p_prefix + '.bn1.bias'

      name_mapping[d_prefix + '_branch2b_w'] = p_prefix + '.conv2.weight'
      name_mapping[d_prefix + '_branch2b_bn_s'] = p_prefix + '.bn2.weight'
      name_mapping[d_prefix + '_branch2b_bn_b'] = p_prefix + '.bn2.bias'

      name_mapping[d_prefix + '_branch2c_w'] = p_prefix + '.conv3.weight'
      name_mapping[d_prefix + '_branch2c_bn_s'] = p_prefix + '.bn3.weight'
      name_mapping[d_prefix + '_branch2c_bn_b'] = p_prefix + '.bn3.bias'

  # RPN
  name_mapping.update({
    'conv_rpn_w': 'RCNN_rpn.RPN_Conv.weight',
    'conv_rpn_b': 'RCNN_rpn.RPN_Conv.bias',
    'rpn_bbox_pred_w': 'RCNN_rpn.RPN_bbox_pred.weight',
    'rpn_bbox_pred_b': 'RCNN_rpn.RPN_bbox_pred.bias',
    'rpn_cls_logits_w': 'RCNN_rpn.RPN_cls_score.weight',
    'rpn_cls_logits_b': 'RCNN_rpn.RPN_cls_score.bias'
  })

  # MASK
  name_mapping.update({
    'conv5_mask_w': 'mask_head.upconv5.weight',
    'conv5_mask_b': 'mask_head.upconv5.bias',
    'mask_fcn_logits_w': 'mask_outputs.classify.weight',
    'mask_fcn_logits_b': 'mask_outputs.classify.bias'
  })

  return name_mapping


def load_detectron_weight(net, detectron_weight_file, map_func):
  with open(detectron_weight_file, 'rb') as fp:
    src_blobs = pickle.load(fp, encoding='latin1')
  if 'blobs' in src_blobs:
    src_blobs = src_blobs['blobs']
  name_mapping = map_func()
  name_mapping = dict((v, k) for k, v in name_mapping.items())
  params = net.state_dict()
  for p_name, p_tensor in params.items():
    if p_name.startswith('mask_head.res5'):
      p_name.replace('mask_head.res5', 'RCNN_top.0')
    d_name = name_mapping[p_name]
    p_tensor.copy_(torch.Tensor(src_blobs[d_name]))


if __name__ == '__main__':
  set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50',
              'RPN.CLS_ACTIVATION', 'sigmoid', 'RPN.OUT_DIM_AS_IN_DIM', 'True']
  cfg_from_list(set_cfgs)
  maskRCNN = resnet([str(n) for n in range(81)], 50, pretrained=True, class_agnostic=False)
  load_detectron_weight(maskRCNN, '/home/roytseng/detectron_models/e2e_mask_rcnn_R-50-C4_2x.pkl')
