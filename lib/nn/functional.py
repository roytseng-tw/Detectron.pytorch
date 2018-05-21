"""Functional interface"""


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    input_shape = x.shape
    ndim = len(input_shape)
    N, C = input_shape[:2]
    G = num_groups
    assert C % G == 0, "input channel dimension must divisible by number of groups"
    x = x.view(N, G, -1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    x = (x - mean) / (var + eps).sqrt()
    x = x.view(input_shape)
    view_shape = (1, -1) + (1,) * (ndim - 2)
    if weight is not None:
        return x * weight.view(view_shape) + bias.view(view_shape)
    return x
