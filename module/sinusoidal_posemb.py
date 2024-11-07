import torch
import torch.nn as nn
import math

"""
Layer:  SinusoidalPoseEmb
Author: cxz21
Data:   2024/10/30
"""


class SinusoidalPosEmb_(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmbFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        device = x.device
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb_sin = emb.sin()
        emb_cos = emb.cos()
        emb = torch.cat((emb_sin, emb_cos), dim=-1)

        ctx.save_for_backward(x, emb, torch.tensor(dim))

        return emb

    @staticmethod
    def backward(ctx, grad_output):
        x, emb, dim = ctx.saved_tensors
        half_dim = dim.item() // 2
        emb_grad = torch.zeros_like(emb)

        emb_grad[:, :half_dim] = grad_output[:, :half_dim] * emb[:, :half_dim].cos()
        emb_grad[:, half_dim:] = -grad_output[:, half_dim:] * emb[:, half_dim:].sin()

        grad_x = emb_grad.sum(dim=-1)
        return grad_x, None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        return SinusoidalPosEmbFunc.apply(x, self.dim)


if __name__ == "__main__":
    dim = 16
    x = torch.randn(4, requires_grad=True)
    y = x

    pos_emb_layer = SinusoidalPosEmb(dim)
    emb = pos_emb_layer(x)

    fake = SinusoidalPosEmb_(dim)
    outy = fake(y)

    print("x output", emb)
    print("y output", outy)

    dout = torch.randn_like(emb)
    fakeloss = (emb * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    print("x grad", x.grad)
    print("y grad", y.grad)
