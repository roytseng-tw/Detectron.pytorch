import torch
import torch.nn as nn


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)
