"""Parameter initialization functions
"""

import math
import operator
from functools import reduce

import torch.nn.init as init


def XavierFill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


def MSRAFill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = math.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)
