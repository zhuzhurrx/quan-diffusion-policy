import torch
from torch import nn


class ManualDropout(nn.Module):
    def __init__(self, p):
        super(ManualDropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = torch.empty_like(x).bernoulli_(1 - self.p) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask
