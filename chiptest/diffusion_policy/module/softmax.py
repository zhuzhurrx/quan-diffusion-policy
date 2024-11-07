import torch
from torch.nn.functional import softmax

"""
Layer:  Softmax
Author: cxz21
Data:   2024/10/10
"""


class SoftmaxFunc(torch.autograd.Function):
    """
    Custom autograd function for Softmax in BF16 precision.
    """

    @staticmethod
    def forward(ctx, input: torch.tensor):
        input = input  # .type(torch.bfloat16)

        x_max = torch.max(input, dim=-1, keepdim=True)[0]
        x_exp = torch.exp(input - x_max)
        output = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
        # for backward
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        input, output = ctx.saved_tensors
        input_grad = output * output_grad
        sum_input_grad = input_grad.sum(dim=-1, keepdim=True)
        input_grad -= output * sum_input_grad
        return input_grad


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return SoftmaxFunc.apply(input)


if __name__ == "__main__":
    batch_size = 224
    X = 10
    Y = 3
    x = torch.randn(batch_size, X, Y, requires_grad=True)  # .type(torch.int8)
    y = x
    out = SoftmaxFunc.apply(x)
    outy = softmax(y, dim=-1)

    print("out: ", out)
    print("outy: ", outy)

    dout = torch.randn(batch_size, X, Y)  # .type(torch.float32)

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    print("dx: ", x.grad)
    print("dy: ", y.grad)
