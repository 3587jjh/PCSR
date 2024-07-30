import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils as mutils
from models import register


class Block(nn.Module):
    def __init__(self, nf, group=1):
        super(Block, self).__init__()
        self.b1 = mutils.EResidualBlock(nf, nf, group=group)
        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


@register('carn')
class CARN_M(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, scale=4, group=4, no_upsampling=False):
        super(CARN_M, self).__init__()
        self.scale = scale
        self.out_dim = nf

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)

        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)

        self.no_upsampling = no_upsampling
        if not no_upsampling:
            self.upsample = mutils.UpsampleBlock(nf, scale=scale, multi_scale=False, group=group)
            self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)
                
    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        out = o3.clone()

        if not self.no_upsampling:
            out = self.upsample(out, scale=self.scale)
            out = self.exit(out)
        #out = self.add_mean(out)
        return out
