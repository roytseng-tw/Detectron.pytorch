import os
import socket
from collections import defaultdict, Iterable
from copy import deepcopy
from datetime import datetime
from itertools import chain

import torch

from core.config import cfg


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


def get_output_dir(args, run_name):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.cfg_file)[1])
    return os.path.join(cfg.OUTPUT_DIR, cfg_filename, run_name)


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
            images.append(os.path.join(dirpath, f))
    return images


def ensure_optimizer_ckpt_params_order(param_groups_names, checkpoint):
    """Reorder the parameter ids in the SGD optimizer checkpoint to match
    the current order in the program, in case parameter insertion order is changed.
    """
    assert len(param_groups_names) == len(checkpoint['optimizer']['param_groups'])
    param_lens = (len(g) for g in param_groups_names)
    saved_lens = (len(g['params']) for g in checkpoint['optimizer']['param_groups'])
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    name_to_curpos = {}
    for i, p_names in enumerate(param_groups_names):
        for j, name in enumerate(p_names):
            name_to_curpos[name] = (i, j)

    param_groups_inds = [[] for _ in range(len(param_groups_names))]
    cnts = [0] * len(param_groups_names)
    for key in checkpoint['model']:
        pos = name_to_curpos.get(key)
        if pos:
            # print(key, pos, cnts[pos[0]])
            saved_p_id = checkpoint['optimizer']['param_groups'][pos[0]]['params'][cnts[pos[0]]]
            assert (checkpoint['model'][key].shape ==
                    checkpoint['optimizer']['state'][saved_p_id]['momentum_buffer'].shape), \
                   ('param and momentum_buffer shape mismatch in checkpoint.'
                    ' param_name: {}, param_id: {}'.format(key, saved_p_id))
            param_groups_inds[pos[0]].append(pos[1])
            cnts[pos[0]] += 1

    for cnt, param_inds in enumerate(param_groups_inds):
        ckpt_params = checkpoint['optimizer']['param_groups'][cnt]['params']
        assert len(ckpt_params) == len(param_inds)
        ckpt_params = [x for x, _ in sorted(zip(ckpt_params, param_inds), key=lambda x: x[1])]
        checkpoint['optimizer']['param_groups'][cnt]['params'] = ckpt_params


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
