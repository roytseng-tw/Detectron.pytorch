import os
import socket
from collections import defaultdict, Iterable
from datetime import datetime
from copy import deepcopy
from itertools import chain

import torch


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


def get_output_dir(args, run_name):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.cfg_file)[1])
    return os.path.join(args.output_base_dir, cfg_filename, run_name)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
      Args:
          filename (string): path to a file
      Returns:
          bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_imagelist_from_dir(dirpath):
    images = []
    for f in os.listdir(dirpath):
        if is_image_file(f):
            images.append(f)
    return images


def load_optimizer_state_dict(optimizer, state_dict):
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
                zip(chain(*(g['params'] for g in saved_groups)),
                    chain(*(g['params'] for g in groups)))}

    def cast(param, value):
        """Make a deep copy of value, casting all tensors to device of param."""
        if torch.is_tensor(value):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if isinstance(param.data, (torch.FloatTensor, torch.cuda.FloatTensor,
                                       torch.DoubleTensor, torch.cuda.DoubleTensor,
                                       torch.HalfTensor, torch.cuda.HalfTensor)):  # param.is_floating_point():
                value = value.type_as(param.data)
            value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group
    param_groups = [
        update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    optimizer.__setstate__({'state': state, 'param_groups': param_groups})
