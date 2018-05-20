"""Functional interface"""


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    N, C, H, W = x.size()
    assert C % num_groups == 0, "input channel dimension must divisible by number of groups"
    x = x.view(N, num_groups, -1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    x = (x - mean) / (var + eps).sqrt()
    x = x.view(N, C, H, W)
    if weight is not None:  # affine=True
        return x * weight + bias
    return x