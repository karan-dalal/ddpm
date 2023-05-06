import torch

from torch import nn
from infrastructure.utils import default, partial, ResnetBlock, SinusoidalPositionEmbeddings, Residual, PreNorm, LinearAttention, Downsample, Attention, Upsample

class Unet(nn.Module):
    """
    The neural network that aims to estimate the noise in a given
    given image. U-Net architecture with attention.

    Contains a convolutional layer, downsampling, attention, upsampling,
    and final convolutional layer.
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
    ):
        super().__init__()

        # Determine dimensions and channels
        self.channels = channels
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # Create different dimensions for each layer
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        # Create various aspects of the U-Net
        self.time_mlp = self.create_time_mlp(dim, time_dim)
        self.downs = self.create_down_layers(in_out, block_klass, time_dim)
        self.mid_block1, self.mid_attn, self.mid_block2 = self.create_mid_layers(dims[-1], block_klass, time_dim)
        self.ups = self.create_up_layers(in_out, block_klass, time_dim)

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def create_time_mlp(self, dim, time_dim):
        return nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def create_down_layers(self, in_out, block_klass, time_dim):
        layers = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        return layers
        
    def create_mid_layers(self, mid_dim, block_klass, time_dim):
        return (
            block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
            Residual(PreNorm(mid_dim, Attention(mid_dim))),
            block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
        )

    def create_up_layers(self, in_out, block_klass, time_dim):
        layers = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            layers.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        return layers

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
