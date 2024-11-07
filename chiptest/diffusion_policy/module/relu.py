import torch

"""
Layer:  ReLU
Author: cxz21
Data:   2024/10/30
"""


def relu(input: torch.Tensor):
    return input.clamp(min=0)


class ReLUFunc(torch.autograd.Function):
    """
    Custom autograd function for Dropout.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        ctx.save_for_backward(input)

        output = input.clamp(min=0)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: torch.Tensor):
        return ReLUFunc.apply(input)


if __name__ == "__main__":
    x = torch.randn(5, 5, requires_grad=True)
    y = x

    out = ReLUFunc.apply(x)
    outy = y.clamp(min=0)

    dout = torch.randn_like(x)
    print("out: ", out)
    print("outy: ", outy)
    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()

    print("dx: ", x.grad)
    print("dy: ", y.grad)
