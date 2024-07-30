import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils as mutils
from models import register

@register('fsrcnn')
class FSRCNN_net(nn.Module):
    def __init__(self, input_channels=3, upscale=4, d=56, s=12, m=4, no_upsampling=False):
        super(FSRCNN_net, self).__init__()
        self.input_channels = input_channels
        self.upscale = upscale
        self.scale = upscale
        self.d = d
        self.s = s
        self.m = m
        self.out_dim = d
        self.no_upsampling = no_upsampling

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        self.body_conv = nn.Sequential(*self.layers)

        # Deconvolution
        if not no_upsampling:
            self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                stride=upscale, padding=3, output_padding=1)
            mutils.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)
        else:
            mutils.initialize_weights([self.head_conv, self.body_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = fea
        if not self.no_upsampling:
            out = self.tail_conv(out)
        return out
