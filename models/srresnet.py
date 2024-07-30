import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils as mutils
from models import register


@register('srresnet-old')
class SRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, no_upsampling=False):
        super(SRResNet, self).__init__()
        self.out_dim = nf
        self.scale = upscale
        self.upscale = upscale
        self.no_upsampling = no_upsampling

        conv = mutils.default_conv
        kernel_size = 3
        act = nn.PReLU

        head = [conv(in_nc, nf, kernel_size=9), act()]
        body = [mutils.ResBlock(conv, nf, kernel_size, bias=True, bn=True, act=act()) for _ in range(nb)]
        body.extend([conv(nf, nf, kernel_size), nn.BatchNorm2d(nf)])
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

        if not no_upsampling:
            tail = [
                mutils.Upsampler(conv, upscale, nf, act=act),
                conv(nf, out_nc, kernel_size)
            ]
            self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        x = x + self.body(x)
        if not self.no_upsampling:
            x = self.tail(x)
        return x


@register('srresnet')
class MSRResNet(nn.Module):
    ''' modified SRResNet'''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, no_upsampling=False):
        super(MSRResNet, self).__init__()
        self.out_dim = nf
        self.scale = upscale
        self.upscale = upscale
        self.no_upsampling = no_upsampling

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutils.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = mutils.make_layer(basic_block, nb)

        if not no_upsampling:
            # upsampling
            if self.upscale == 2:
                self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
                self.pixel_shuffle = nn.PixelShuffle(2)
            elif self.upscale == 3:
                self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
                self.pixel_shuffle = nn.PixelShuffle(3)
            elif self.upscale == 4:
                self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
                self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
                self.pixel_shuffle = nn.PixelShuffle(2)

            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

            # initialization
            mutils.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
            if self.upscale == 4:
                mutils.initialize_weights(self.upconv2, 0.1)
        else:
            mutils.initialize_weights(self.conv_first, 0.1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if not self.no_upsampling:
            if self.upscale == 4:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            elif self.upscale == 3 or self.upscale == 2:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

            out = self.conv_last(self.lrelu(self.HRconv(out)))
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            out += base
        return out
