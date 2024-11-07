import copy
from typing import Optional

import torch
import torch.nn as nn
from layernorm import LayerNorm
from linear import Linear
from multihead_attn import MultiHeadAttention
from relu import relu
from torch.nn.modules.container import ModuleList
from dropout import ManualDropout
from torch.nn.modules.transformer import TransformerDecoderLayer


class TransformerDecoderLayer_(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ) -> None:
        super(TransformerDecoderLayer_, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout_p=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout_p=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = ManualDropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.dropout1 = ManualDropout(dropout)
        self.dropout2 = ManualDropout(dropout)
        self.dropout3 = ManualDropout(dropout)

        self.activation = relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x, atten_mask=attn_mask)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self, x: torch.Tensor, mem: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.multihead_attn(x, mem, mem, atten_mask=attn_mask)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for _ in range(N)])


if __name__ == "__main__":
    # parameter
    n_emb = 256
    n_head = 4
    n_layer = 5
    dropout_p = 0.0
    decoder_layer = TransformerDecoderLayer(
        d_model=n_emb,
        nhead=n_head,
        dim_feedforward=4 * n_emb,
        dropout=dropout_p,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )
    fake_layer = TransformerDecoderLayer_(
        d_model=n_emb,
        nhead=n_head,
        dim_feedforward=4 * n_emb,
        dropout=dropout_p,
        norm_first=True,
    )
    in_proj_weight1 = nn.Parameter(torch.randn(3 * n_emb, n_emb, requires_grad=True))
    out_proj_weight1 = nn.Parameter(torch.randn(n_emb, n_emb, requires_grad=True))
    in_proj_weight2 = nn.Parameter(torch.randn(3 * n_emb, n_emb, requires_grad=True))
    out_proj_weight2 = nn.Parameter(torch.randn(n_emb, n_emb, requires_grad=True))
    linear1_weight = nn.Parameter(torch.randn(4 * n_emb, n_emb))
    linear2_weight = nn.Parameter(torch.randn(n_emb, 4 * n_emb))
    # decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

    # input
    tgt = torch.randn(56, 10, 256, requires_grad=True)
    memory = torch.randn(56, 3, 256, requires_grad=True)
    fake_tgt = tgt
    fake_memory = memory

    T = 10
    S = 3
    sz = T
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
    memory_mask = t >= (s - 1)
    memory_mask = (
        memory_mask.float()
        .masked_fill(memory_mask == 0, float("-inf"))
        .masked_fill(memory_mask == 1, float(0.0))
    )
    # print(mask)
    # print(memory_mask)
    decoder_layer.linear1.weight = linear1_weight
    fake_layer.linear1.weight = linear1_weight
    decoder_layer.linear2.weight = linear2_weight
    fake_layer.linear2.weight = linear2_weight
    decoder_layer.self_attn.in_proj_weight = nn.Parameter(in_proj_weight1)
    fake_layer.self_attn.queries.weight = nn.Parameter(in_proj_weight1[0:n_emb, :])
    fake_layer.self_attn.keys.weight = nn.Parameter(
        in_proj_weight1[n_emb : 2 * n_emb, :]
    )
    fake_layer.self_attn.values.weight = nn.Parameter(
        in_proj_weight1[2 * n_emb : 3 * n_emb, :]
    )
    decoder_layer.multihead_attn.in_proj_weight = nn.Parameter(in_proj_weight2)
    fake_layer.multihead_attn.queries.weight = nn.Parameter(in_proj_weight2[0:n_emb, :])
    fake_layer.multihead_attn.keys.weight = nn.Parameter(
        in_proj_weight2[n_emb : 2 * n_emb, :]
    )
    fake_layer.multihead_attn.values.weight = nn.Parameter(
        in_proj_weight2[2 * n_emb : 3 * n_emb, :]
    )
    decoder_layer.self_attn.out_proj.weight = nn.Parameter(out_proj_weight1)
    fake_layer.self_attn.fc_out.weight = nn.Parameter(out_proj_weight1)
    decoder_layer.multihead_attn.out_proj.weight = nn.Parameter(out_proj_weight2)
    fake_layer.multihead_attn.fc_out.weight = nn.Parameter(out_proj_weight2)
    out = decoder_layer(0, tgt, memory, mask, memory_mask)
    dout = torch.randn_like(out)
    loss = (out * dout).sum()
    loss.backward()
    fake_out = fake_layer(fake_tgt, fake_memory, mask, memory_mask)
    fake_loss = (fake_out * dout).sum()
    fake_loss.backward()

    print(out)
    print(fake_out)
    print(out - fake_out)

    # print(tgt.grad)
    # print(memory.grad)
    # print(fake_tgt.grad)
    # print(fake_memory.grad)
