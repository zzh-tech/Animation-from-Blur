import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import SameBlock2d, ResBlock2d, UpBlock2d, DownBlock2d, UpBlockTrans2d, DownBlockConv2d, RDNet


class Decomposer(nn.Module):
    """
    Basic decomposer to extract sharp image sequence from a blurry image
    """

    def __init__(self, in_channels, out_channels, block_expansion, num_down_blocks, num_bottleneck_blocks,
                 max_features, conv_down=False, trans_up=False, rdb=False, rdb_args=None, use_norm=True,
                 norm_type='batch', injection=False):
        super(Decomposer, self).__init__()
        self.injection = injection
        self.first_conv = SameBlock2d(in_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3),
                                      use_norm=use_norm, norm_type=norm_type)

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            if self.injection:
                in_features += 2
            if conv_down:
                down_blocks.append(
                    DownBlockConv2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), use_norm=use_norm,
                                    norm_type=norm_type))
            else:
                down_blocks.append(
                    DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), use_norm=use_norm,
                                norm_type=norm_type))
        self.down_blocks = nn.ModuleList(down_blocks)

        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        if rdb:
            self.bottleneck = RDNet(in_chs=in_features, num_blocks=num_bottleneck_blocks, **rdb_args)
        else:
            bottleneck = []
            for i in range(num_bottleneck_blocks):
                bottleneck.append(
                    ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1), use_norm=use_norm, norm_type=norm_type))
            self.bottleneck = nn.Sequential(*bottleneck)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            if trans_up:
                up_blocks.append(
                    UpBlockTrans2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), use_norm=use_norm,
                                   norm_type=norm_type))
            else:
                up_blocks.append(
                    UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), use_norm=use_norm,
                              norm_type=norm_type))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(block_expansion, out_channels, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: blurry image (8, 3, 384, 288)
        :return: shape image sequence (8, num_gts, 3, 384, 288)
        """
        b, c, h, w = x.shape
        trend = x[:, -2:]
        out = self.first_conv(x)
        for i in range(len(self.down_blocks)):
            if self.injection:
                trend = F.interpolate(trend, size=(h // (2 ** i), w // (2 ** i)), mode='bilinear')
                out = self.down_blocks[i](torch.cat([out, trend], dim=1))
            else:
                out = self.down_blocks[i](out)
        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final_conv(out)
        out = out.reshape(b, out.size(1) // 3, 3, h, w)
        return out


if __name__ == '__main__':
    x = torch.randn(4, 3, 384, 288).cuda()
    model = Decomposer(in_channels=3,
                       out_channels=21,
                       block_expansion=64,
                       num_down_blocks=2,
                       num_bottleneck_blocks=6,
                       max_features=512).cuda()
    out = model(x)
    print(out.shape)
