import torch
import torch.nn as nn
import torch.nn.functional as F
from linear import Linear
from torch.autograd import Function


class MultiHeadAttentionFunc(Function):
    @staticmethod
    def forward(ctx, queries, keys, values, num_heads, dropout_p, atten_mask):
        N, value_len, embed_size = values.shape
        _, key_len, _ = keys.shape
        _, query_len, _ = queries.shape

        head_dim = embed_size // num_heads

        values = values.reshape(N, value_len, num_heads, head_dim)
        keys = keys.reshape(N, key_len, num_heads, head_dim)
        queries = queries.reshape(N, query_len, num_heads, head_dim)

        # print(
        #     queries.transpose(1, 2).reshape(224, 10, 64) / (embed_size ** (1 / 2)) * 2
        # )
        # print(keys)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # print(energy.size())
        # print(energy.reshape(224, 10, 10))
        if atten_mask is not None:
            energy = energy.masked_fill(atten_mask == float("-inf"), float("-inf"))

        # print(energy.size())
        # print((energy / (embed_size ** (1 / 2))).reshape(224, 10, 10))
        attention = F.softmax(energy / (head_dim ** (1 / 2)), dim=3)

        if dropout_p:
            attention = F.dropout(attention, p=dropout_p)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, embed_size
        )

        ctx.save_for_backward(queries, keys, values, attention, atten_mask)
        ctx.num_heads = num_heads
        ctx.dropout_p = dropout_p
        return out

    @staticmethod
    def backward(ctx, grad_output):
        queries, _, values, attention, _ = ctx.saved_tensors
        num_heads = ctx.num_heads

        N, query_len, embed_size = grad_output.shape
        head_dim = embed_size // num_heads

        grad_values = grad_keys = grad_queries = None

        if ctx.needs_input_grad[0]:
            grad_attention = torch.einsum(
                "nqhd,nlhd->nhql",
                grad_output.reshape(N, query_len, num_heads, head_dim),
                values,
            )
            grad_attention = (
                grad_attention * (attention * (1 - attention)) / (embed_size ** (1 / 2))
            )
            grad_values = torch.einsum("nhql,nqhd->nlhd", grad_attention, queries)
            grad_values = grad_values.reshape(N, -1, embed_size)

        if ctx.needs_input_grad[1]:
            grad_keys = torch.einsum("nhql,nqhd->nlhd", grad_attention, queries)
            grad_keys = grad_keys.reshape(N, -1, embed_size)

        if ctx.needs_input_grad[2]:
            grad_queries = torch.einsum("nhql,nlhd->nqhd", grad_attention, values)
            grad_queries = grad_queries.reshape(N, -1, embed_size)

        return grad_queries, grad_keys, grad_values, None, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.dropout_p = dropout_p

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = Linear(embed_size, embed_size, bias=False)
        self.keys = Linear(embed_size, embed_size, bias=False)
        self.queries = Linear(embed_size, embed_size, bias=False)
        self.fc_out = Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, atten_mask=None):
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        out = MultiHeadAttentionFunc.apply(
            queries, keys, values, self.num_heads, self.dropout_p, atten_mask
        )
        out = self.fc_out(out)
        return out


if __name__ == "__main__":
    embed_size = 256
    num_heads = 4
    dropout_p = 0.0
    batch_size = 56
    sequence_length = 10

    x = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y = x

    in_proj_weight = nn.Parameter(
        torch.randn(3 * embed_size, embed_size, requires_grad=True)
    )
    out_proj_weight = nn.Parameter(
        torch.randn(embed_size, embed_size, requires_grad=True)
    )

    atten_mask = torch.ones(sequence_length, sequence_length)

    out = F.multi_head_attention_forward(
        query=x.transpose(1, 0),
        key=x.transpose(1, 0),
        value=x.transpose(1, 0),
        embed_dim_to_check=embed_size,
        num_heads=num_heads,
        in_proj_weight=in_proj_weight,
        in_proj_bias=None,
        attn_mask=atten_mask,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=dropout_p,
        out_proj_weight=out_proj_weight,
        out_proj_bias=None,
    )[0]

    dout = torch.randn_like(out)
    loss = (out * dout).sum()
    loss.backward()

    attn_layer = MultiHeadAttention(
        embed_size=embed_size, num_heads=num_heads, dropout_p=dropout_p
    )
    attn_layer.queries.weight = nn.Parameter(in_proj_weight[0:embed_size, :])
    attn_layer.keys.weight = nn.Parameter(
        in_proj_weight[embed_size : 2 * embed_size, :]
    )
    attn_layer.values.weight = nn.Parameter(
        in_proj_weight[2 * embed_size : 3 * embed_size, :]
    )
    attn_layer.fc_out.weight = nn.Parameter(out_proj_weight)
    attn_layer.queries.bias = None
    attn_layer.keys.bias = None
    attn_layer.values.bias = None
    attn_layer.fc_out.bias = None
    fake_out = attn_layer(y, y, y, atten_mask).transpose(0, 1)

    print("out:\n", out)
    print("fake_out:\n", fake_out)
    print(out - fake_out)
    print("Gradient of input:\n", x.grad)
    print("Gradient of input:\n", y.grad)
    print(x.grad - y.grad)
