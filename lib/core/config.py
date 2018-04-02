from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
import yaml

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = AttrDict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Trim size for input images to create minibatch #TODO: deprecate
__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600

# Images *per GPU* in the training minibatch
# Total images per minibatch = TRAIN.IMS_PER_BATCH * NUM_GPUS
__C.TRAIN.IMS_PER_BATCH = 2

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
__C.TRAIN.BATCH_SIZE_PER_IM = 64

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
# __C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True) (legacy)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide)
# This feature is critical for saving memory (and makes training slightly
# faster)
__C.TRAIN.ASPECT_GROUPING = True

# Crop images that have too small or too large aspect ratio
__C.TRAIN.ASPECT_CROPPING = False

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples (deprecated)
__C.TRAIN.RPN_BATCHSIZE = 256
# Total number of RPN examples per image
__C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.TRAIN.RPN_STRADDLE_THRESH = 0

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

# Whether to tune the batch normalization parameters during training
__C.TRAIN.BN_TRAIN = False

# Filter proposals that are inside of crowd regions by CROWD_FILTER_THRESH
# "Inside" is measured as: proposal-with-crowd intersection area divided by
# proposal area
__C.TRAIN.CROWD_FILTER_THRESH = 0.7

# Ignore ground-truth objects with area < this threshold
__C.TRAIN.GT_MIN_AREA = -1

#
# Testing options
#
__C.TEST = AttrDict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 1000

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.TEST.SCORE_THRESH = 0.05

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.TEST.DETECTIONS_PER_IM = 100

# ---------------------------------------------------------------------------- #
# Soft NMS
# ---------------------------------------------------------------------------- #
__C.TEST.SOFT_NMS = AttrDict()

# Use soft NMS instead of standard NMS if set to True
__C.TEST.SOFT_NMS.ENABLED = False
# See soft NMS paper for definition of these options
__C.TEST.SOFT_NMS.METHOD = 'linear'
__C.TEST.SOFT_NMS.SIGMA = 0.5
# For the soft NMS overlap threshold, we simply use TEST.NMS

# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_VOTE = AttrDict()

# Use box voting if set to True
__C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
__C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0


#
# MobileNet options
#

__C.MOBILENET = AttrDict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the first of all 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 16.

# Clip bounding box transformation predictions to prevent np.exp from
# overflowing
# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# "Fun" fact: the history of where these values comes from is lost (From Detectron lol)
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 20

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8, 16, 32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5, 1, 2]

# Feature stride for RPN
__C.FEAT_STRIDE = [
    16,
]

__C.CUDA = False

__C.CROP_RESIZE_WITH_MAX_POOL = True

# Train or Inference. This affect the model construction !!!
__C.IS_TRAIN = True

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The backbone conv body to use
__C.MODEL.CONV_BODY = ''

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# Use a class agnostic bounding box regressor instead of the default per-class
# regressor
__C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
#
# In older versions, the weights were set such that the regression deltas
# would have unit standard deviation on the training dataset. Presently, rather
# than computing these statistics exactly, we use a fixed set of weights
# (10., 10., 5., 5.) by default. These are approximately the weights one would
# get from COCO using the previous unit stdev heuristic.
__C.MODEL.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# The meaning of FASTER_RCNN depends on the context (training vs. inference):
# 1) During training, FASTER_RCNN = True means that end-to-end training will be
#    used to jointly train the RPN subnetwork and the Fast R-CNN subnetwork
#    (Faster R-CNN = RPN + Fast R-CNN).
# 2) During inference, FASTER_RCNN = True means that the model's RPN subnetwork
#    will be used to generate proposals rather than relying on precomputed
#    proposals. Note that FASTER_RCNN = True can be used at inference time even
#    if the Faster R-CNN model was trained with stagewise training (which
#    consists of alternating between RPN and Fast R-CNN training in a way that
#    finally leads to a single network).
__C.MODEL.FASTER_RCNN = False

# Indicates the model makes instance mask predictions (as in Mask R-CNN)
__C.MODEL.MASK_ON = False

# Indicates the model makes keypoint predictions (as in Mask R-CNN for
# keypoints)
__C.MODEL.KEYPOINTS_ON = False

# Indicates the model's computation terminates with the production of RPN
# proposals (i.e., it outputs proposals ONLY, no actual object detections)
__C.MODEL.RPN_ONLY = False

# Indicate whether the res5 stage forward computation is shared or not on training
__C.MODEL.SHARE_RES5 = False

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()

# `True` for Detectron implementation. `False` for jwyang's implementation.
__C.RPN.OUT_DIM_AS_IN_DIM = True

# Output dim of conv2d. Ignored if `__C.RPN.OUT_DIM_AS_IN_DIM` is True.
# 512 is the fixed value in jwyang's implementation.
__C.RPN.OUT_DIM = 512

# 'sigmoid' or 'softmax'. Detectron use 'sigmoid'. jwyang use 'softmax'
# This will affect the conv2d output dim for classifying the bg/fg rois
__C.RPN.CLS_ACTIVATION = 'sigmoid'

# RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
# Note: these options are *not* used by FPN RPN; see FPN.RPN* options
__C.RPN.SIZES = (64, 128, 256, 512)

# Stride of the feature map that RPN is attached
__C.RPN.STRIDE = 16

# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1, 2)


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #

__C.FPN = AttrDict()

# FPN is enabled if True
__C.FPN.FPN_ON = False

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# Initialize the lateral connections to output zero if True
__C.FPN.ZERO_INIT_LATERAL = False

# Stride of the coarsest FPN level
# This is needed so the input can be padded properly
__C.FPN.COARSEST_STRIDE = 32

#
# FPN may be used for just RPN, just object detection, or both
#

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = False
# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0
__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5
# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = False
# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6
# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2
# FPN RPN anchor aspect ratios
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this size on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
__C.FPN.RPN_ANCHOR_START_SIZE = 32
# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()

# The type of RoI head to use for bounding box classification and regression
# The string must match a function this is imported in modeling.model_builder
# (e.g., 'head_builder.add_roi_2mlp_head' to specify a two hidden layer MLP)
__C.FAST_RCNN.ROI_BOX_HEAD = ''

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.MLP_HEAD_DIM = 1024

# RoI transformation function (e.g., RoIPool or RoIAlign)
# (RoIPoolF is the same as RoIPool; ignore the trailing 'F')
__C.FAST_RCNN.ROI_XFORM_METHOD = b'RoIPoolF'

# Number of grid sampling points in RoIAlign (usually use 2)
# Only applies to RoIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transform output resolution
# Note: some models may have constraints on what they can use, e.g. they use
# pretrained FC layers like in VGG16, and will ignore this option
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14

# ---------------------------------------------------------------------------- #
# Mask R-CNN options ("MRCNN" means Mask R-CNN)
# ---------------------------------------------------------------------------- #
__C.MRCNN = AttrDict()

__C.MRCNN.ROI_MASK_HEAD = ''

__C.MRCNN.MEMORY_EFFICIENT_LOSS = True

# Resolution of mask predictions
__C.MRCNN.RESOLUTION = 14

# Number of channels in the mask head
__C.MRCNN.DIM_REDUCED = 256

# Use dilated convolution in the mask head
__C.MRCNN.DILATION = 2

# Upsample the predicted masks by this factor
__C.MRCNN.UPSAMPLE_RATIO = 1

# Weight initialization method for the mask head and mask output layers. ['GaussianFill', 'MSRAFill']
__C.MRCNN.CONV_INIT = 'GaussianFill'

# Use class specific mask predictions if True (otherwise use class agnostic mask
# predictions)
__C.MRCNN.CLS_SPECIFIC_MASK = True

# Binarization threshold for converting soft masks to hard masks
__C.MRCNN.THRESH_BINARIZE = 0.5

# ---------------------------------------------------------------------------- #
# Keyoint Mask R-CNN options ("KRCNN" = Mask R-CNN with Keypoint support)
# ---------------------------------------------------------------------------- #
__C.KRCNN = AttrDict()

__C.KRCNN.ROI_KEYPOINTS_HEAD = ''

# Number of keypoints in the dataset (e.g., 17 for COCO)
__C.KRCNN.NUM_KEYPOINTS = -1

# Number of stacked Conv layers in keypoint head
__C.KRCNN.NUM_STACKED_CONVS = 8

# Output size (and size loss is computed on), e.g., 56x56
__C.KRCNN.HEATMAP_SIZE = -1

# ---------------------------------------------------------------------------- #
# ResNets options ("ResNets" = ResNet and ResNeXt)
# ---------------------------------------------------------------------------- #
__C.RESNETS = AttrDict()

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
__C.RESNETS.STRIDE_1X1 = True

# Apply dilation in stage "res5"
__C.RESNETS.RES5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.RESNETS.FREEZE_AT = 2

# Whether to load pretrained weight from ImageNet
__C.RESNETS.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# Deprecated options (old option from jwyang)
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set((
    # 'ANCHOR_SCALES',
    # 'ANCHOR_RATIOS',
    # 'FEAT_STRIDE'
))

# ---------------------------------------------------------------------------- #
# Renamed options (old option from jwyang)
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'TRAIN.BATCH_SIZE': 'TRAIN.BATCH_SIZE_PER_IM'
}


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a

