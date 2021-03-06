from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: This is a modified version of OpenAI's unet-ipml which can be found at:
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def normalize(num_channels):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)

def nonlinearity(inplace=False):
    return nn.SiLU(inplace=inplace)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = conv3x3(channels, channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = conv3x3(channels, channels, stride=2)
        else:
            self.op = nn.AvgPool2d(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, 
        in_channels,
        emb_channels,
        dropout,
        out_channels = None,
        use_scale_shift_norm = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalize(in_channels),
            nonlinearity(),
            conv3x3(in_channels, self.out_channels)
        )
        self.emb_layers = nn.Sequential(
            nonlinearity(inplace=False),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )
        self.out_layers = nn.Sequential(
            normalize(self.out_channels),
            nonlinearity(),
            nn.Dropout(p=dropout),
            zero_module(
                conv3x3(self.out_channels, self.out_channels)
            )
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv1x1(in_channels, self.out_channels)

    def forward(self, x, t_embed):
        h = self.in_layers(x)

        emb_out = self.emb_layers(t_embed)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalize(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        
      
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class UNetDiffusionModel(nn.Module):
    def __init__(self, 
        in_channels=2,
        out_channels=2,
        model_channels=128,
        num_res_blocks=3,
        channel_mult=(1, 2, 3, 4),
        dropout=0,
        use_scale_shift_norm=True,
        attention_resolutions=(2,4),
        num_heads=4,
        num_heads_upsample=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nonlinearity(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv3x3(in_channels, model_channels))
            ])

        input_block_chans = [model_channels]
        ch = model_channels

        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=True))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalize(ch),
            nonlinearity(),
            zero_module(conv3x3(model_channels, out_channels)),
        )

    def timestep_embedding(self, dim, t, stretch=5000, max_period=10000.0):
        half = dim // 2
        div_term = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device) / half) * stretch
        pe = torch.cat([torch.sin(t * div_term), torch.cos(t * div_term)], dim=-1)
        return pe

    def forward(self, x, t):
        # get positional embedding for timestep
        emb = self.time_embed(self.timestep_embedding(self.model_channels, t))

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)


def main():
    model = UNetDiffusionModel(in_channels=2, out_channels=2, model_channels=128, num_res_blocks=3, channel_mult=(1,2), dropout=0.1)

    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # test batch
    n,c,h,w = 5,2,32,32
    x = torch.randn(n, c, h, w)
    t = torch.rand(n, 1)
    
    print('x', x.size())
    y = model.forward(x, t)
    print('y', x.size())

    y.sum().backward()
    

if __name__ == '__main__':
    main()
