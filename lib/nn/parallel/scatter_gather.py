import collections
import re
import numpy as np
import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import numpy_type_map


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    error_msg = "outputs must contain tensors, numbers, dicts or lists; found {}"

    def gather_map(outputs):
        out = outputs[0]
        elem_type = type(out)
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
        elif isinstance(out, collections.Mapping):
            return {key: gather_map([d[key] for d in outputs]) for key in out}
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = out
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return Variable(torch.from_numpy(np.concatenate(outputs, dim)))
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return Variable(numpy_type_map[elem.dtype.name](list(map(py_type, outputs))))
        elif isinstance(out, int_classes):
            return Variable(torch.LongTensor(outputs))
        elif isinstance(out, float):
            return Variable(torch.DoubleTensor(outputs))
        elif isinstance(out, string_classes):
            return outputs

        raise TypeError((error_msg.format(elem_type)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None
