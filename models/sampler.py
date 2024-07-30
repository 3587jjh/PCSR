import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif-sampler')
class LIIF_Sampler(nn.Module):
    # feature unfolding, local ensemble not supported
    def __init__(self, imnet_spec, in_dim, out_dim):
        super().__init__()
        self.imnet = models.make(imnet_spec, args={'in_dim': in_dim+4, 'out_dim': out_dim})

    def make_inp(self, feat, coord, cell):
        feat_coord = make_coord(feat.shape[-2:], flatten=False, device=feat.device)\
            .permute(2,0,1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        q_feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest',
            align_corners=False)[:,:,0,:].permute(0,2,1)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest',
            align_corners=False)[:,:,0,:].permute(0,2,1)

        rel_coord = coord - q_coord
        rel_coord[:,:,0] *= feat.shape[-2]
        rel_coord[:,:,1] *= feat.shape[-1]

        rel_cell = cell.clone()
        rel_cell[:,:,0] *= feat.shape[-2]
        rel_cell[:,:,1] *= feat.shape[-1]

        inp = torch.cat([q_feat, rel_coord, rel_cell], dim=-1)
        return inp    

    def forward(self, x, coord=None, cell=None):
        if coord is not None:
            x = self.make_inp(x, coord, cell)
        x = self.imnet(x)
        return x