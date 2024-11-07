import torch

"""
Layer:  LayerNorm
Author: shuyuan-19
Data:   2024/10/30
"""


class LayerNormFunc(torch.autograd.Function):
    """
    Custom autograd function for LayerNorm in BF16 precision.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.tensor,
        weight=None,
        bias=None,
        eps=1e-5,
    ):
        input = input

        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True)
        std = torch.sqrt(var + eps)
        normlized_x = (input - mean) / std
        if weight is not None:
            if bias is not None:
                output = normlized_x * weight + bias
            else:
                output = normlized_x * weight
        else:
            output = normlized_x

        # for backward
        ctx.save_for_backward(input, weight, mean, std)

        return output

    @staticmethod
    def backward(ctx, dout):  #: torch.bfloat16
        input, weight, mean, std = ctx.saved_tensors
        dout = dout  # .type(torch.bfloat16)

        norm = (input - mean) / std
        if weight is not None:
            dnorm = dout * weight
        else:
            dnorm = dout

        d_input = (
            dnorm
            - dnorm.mean(dim=-1, keepdim=True)
            - norm * (dnorm * norm).mean(dim=-1, keepdim=True)
        )
        d_input = d_input / std

        return d_input, None, None, None


class LayerNorm(torch.nn.Module):
    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        super(LayerNorm, self).__init__()

    def forward(self, input: torch.Tensor):
        return LayerNormFunc.apply(input, self.weight, self.bias, self.eps)


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    B = 1
    T = 2
    C = 10
    x = torch.randn(B, T, C, requires_grad=True)
    y = x
    w = torch.ones(C, requires_grad=True)
    b = torch.zeros(C, requires_grad=True)
    out = LayerNormFunc.apply(x, w, b)

    dout = torch.randn(B, T, C)

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    mean = torch.mean(y, dim=-1, keepdim=True)
    var = torch.var(y, dim=-1, keepdim=True)
    std = torch.sqrt(var + 1e-5)
    norm = (y - mean) / std
    wy = torch.ones(C, requires_grad=True)
    by = torch.zeros(C, requires_grad=True)
    outy = norm * wy + by
    print("out: ", out)
    print("outy: ", outy)
    print(out - outy)

    loss = (outy * dout).sum()
    loss.backward()
    print("dx: ", x.grad)
    print("dy: ", y.grad)
