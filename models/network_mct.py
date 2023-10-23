## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor,outdim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2,  kernel_size=3, stride=1, padding=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, outdim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3,  kernel_size=3, stride=1, padding=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim,  kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor,outdim, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor,outdim, bias)
        self.patch_embed = OverlapPatchEmbed(dim, outdim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.patch_embed(x) + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class MCT(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=[48, 48, 24, 24, 24, 12, 12, 8, 4],
                 heads=[8, 8, 6, 6, 6, 4, 4, 2, 1],
                 ffn_expansion_factor=3,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'

                 ):

        super(MCT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])

        self.feat1 = TransformerBlock(dim=dim[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,outdim=dim[1], bias=bias,
                             LayerNorm_type=LayerNorm_type)
        self.feat2 = TransformerBlock(dim=dim[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[2], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat3 = TransformerBlock(dim=dim[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[3], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat4 = TransformerBlock(dim=dim[3], num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[4], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat5 = TransformerBlock(dim=dim[4], num_heads=heads[4], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[5], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat6 = TransformerBlock(dim=dim[5], num_heads=heads[5], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[6], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat7 = TransformerBlock(dim=dim[6], num_heads=heads[6], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[7], bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.feat8 = TransformerBlock(dim=dim[7], num_heads=heads[7], ffn_expansion_factor=ffn_expansion_factor,
                                      outdim=dim[8], bias=bias,
                                      LayerNorm_type=LayerNorm_type)

        self.sum = OverlapPatchEmbed(156, 48)
        self.final = OverlapPatchEmbed(24, 3)
        self.upsampler = Upsample(48)



    def forward(self, inp_img):

        x0 = self.patch_embed(inp_img)
        x1 = self.feat1(x0)
        x2 = self.feat2(x1)
        x3 = self.feat3(x2)
        x4 = self.feat4(x3)
        x5 = self.feat5(x4)
        x6 = self.feat6(x5)
        x7 = self.feat7(x6)
        x8 = self.feat8(x7)

        x_sum = self.sum(torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],1))
        x=self.upsampler(x_sum + x0)



        return self.final(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import time
if __name__ == '__main__':
    start_time = time.time()
    upscale = 2
    window_size = 7
    height = 120
    width = 200


    model = MCT()
    pretrained_model = torch.load('../model_zoo/115000_G.pth')
    model.load_state_dict(pretrained_model)


    x = torch.ones((1, 3, height, width))
    print(x.shape)
    y = model(x)
    print(count_parameters(model))

    # print(model.flops())
    print(y.shape)
    print("--- %s seconds ---" % (time.time() - start_time))
