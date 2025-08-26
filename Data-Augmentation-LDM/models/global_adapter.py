import torch
from torch import nn
from einops import rearrange

from ldm.modules.attention import FeedForward


class GlobalAdapter(nn.Module):
    def __init__(self, in_dim, channel_mult=[2, 4]):
        super().__init__()
        dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
        dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
        self.in_dim = in_dim
        self.channel_mult = channel_mult
        
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.1)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)

        self.linear = nn.Linear(4096, 2048)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.squeeze(x, 1)

        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))

        if x.shape[1] == 4096:
            x = self.linear(x)

        x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=512).contiguous()
        return x