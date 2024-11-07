import torch

"""
Layer:  Linear
Author: cxz21
Data:   2024/10/30
"""


class LinearFunc(torch.autograd.Function):
    """
    Custom autograd function for Linear in int8 precision.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias=None):
        input = input  # .type(torch.int8)
        weight = weight  # .type(torch.int8)
        if bias is not None:
            bias = bias  # .type(torch.int8)

        if bias is not None:
            output = input @ weight.transpose(0, 1) + bias[None, ...]
        else:
            output = input @ weight.transpose(0, 1)

        ctx.save_for_backward(input, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_output = grad_output  # .type(torch.int8)

        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output @ weight
        grad_weight = grad_output.transpose(-2, -1) @ input
        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class Linear(torch.nn.Linear):
    def __init__(self, in_feature: int, out_feature: int, bias: bool = False):
        super().__init__(in_feature, out_feature, bias)

    def forward(self, input):
        return LinearFunc.apply(input, self.weight, self.bias)


if __name__ == "__main__":
    in_features = 2
    batch_size = 10
    out_features = 5
    x = torch.randn(batch_size, in_features, requires_grad=True)  # .type(torch.int8)
    y = x
    w = torch.ones(out_features, in_features, requires_grad=True)  # .type(torch.int8)
    b = torch.zeros(out_features, requires_grad=True)  # .type(torch.int8)

    out = LinearFunc.apply(x, w, b)
    outy = y.matmul(w.transpose(0, 1)) + b  # .type(torch.float32)

    print("out: ", out)
    print("outy: ", outy)

    dout = torch.randn(batch_size, out_features)  # .type(torch.float32)

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    print("dx: ", x.grad)
    print("dy: ", y.grad)
