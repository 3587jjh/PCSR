import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from fast_pytorch_kmeans import KMeans
from utils import *
from flops import get_model_flops


@register('pcsr-phase0')
class PCSR(nn.Module):
    def __init__(self, encoder_spec, heavy_sampler_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        in_dim = self.encoder.out_dim
        self.heavy_sampler = models.make(heavy_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})

    def forward(self, lr, coord, cell, **kwargs):
        if self.training:
            return self.forward_train(lr, coord, cell)
        else:
            return self.forward_test(lr, coord, cell, **kwargs)

    def forward_train(self, lr, coord, cell):
        feat = self.encoder(lr)
        res = F.grid_sample(lr, coord.flip(-1).unsqueeze(1), mode='bilinear', 
            padding_mode='border', align_corners=False)[:,:,0,:].permute(0,2,1)
        pred_heavy = self.heavy_sampler(feat, coord, cell) + res
        return pred_heavy

    def forward_test(self, lr, coord, cell, pixel_batch_size=None):
        feat = self.encoder(lr)
        b,q = coord.shape[:2]
        tot = b*q
        if not pixel_batch_size:
            pixel_batch_size = q

        preds = []
        for i in range(b): # for each image
            pred = torch.zeros((q,3), device=lr.device)
            l = 0
            while l < q:
                r = min(q, l+pixel_batch_size)
                coord_split = coord[i:i+1,l:r,:]
                cell_split = cell[i:i+1,l:r,:]
                res = F.grid_sample(lr[i:i+1], coord_split.flip(-1).unsqueeze(1), mode='bilinear', 
                    padding_mode='border', align_corners=False)[:,:,0,:].squeeze(0).transpose(0,1)
                pred[l:r] = self.heavy_sampler(feat[i:i+1], coord_split, cell_split) + res
                l = r
            preds.append(pred)
        pred = torch.stack(preds, dim=0)
        return pred


@register('pcsr-phase1')
class PCSR(nn.Module):

    def __init__(self, encoder_spec, heavy_sampler_spec, light_sampler_spec, classifier_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        in_dim = self.encoder.out_dim
        self.heavy_sampler = models.make(heavy_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})
        self.light_sampler = models.make(light_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})
        self.classifier = models.make(classifier_spec,
            args={'in_dim': in_dim, 'out_dim': 2})
        self.kmeans = KMeans(n_clusters=2, max_iter=20, mode='euclidean', verbose=0)
        self.cost_list = {}

    def forward(self, lr, coord, cell, **kwargs):
        if self.training:
            return self.forward_train(lr, coord, cell)
        else:
            return self.forward_test(lr, coord, cell, **kwargs)

    def forward_train(self, lr, coord, cell):
        feat = self.encoder(lr)
        prob = self.classifier(feat, coord, cell)
        prob = F.softmax(prob, dim=-1) # (b,q,2)

        pred_heavy = self.heavy_sampler(feat, coord, cell)
        pred_light = self.light_sampler(feat, coord, cell)
        pred = prob[:,:,0:1] * pred_light + prob[:,:,1:2] * pred_heavy

        res = F.grid_sample(lr, coord.flip(-1).unsqueeze(1), mode='bilinear', 
            padding_mode='border', align_corners=False)[:,:,0,:].permute(0,2,1)
        pred = pred + res
        return pred, prob

    def forward_test(self, lr, coord, cell, scale=None, hr_size=None, k=0., pixel_batch_size=None, adaptive_cluster=False, refinement=True):
        h,w = lr.shape[-2:]
        if not scale and hr_size:
            H,W = hr_size
            scale = round((H/h + W/w)/2, 1)
        else:
            assert scale and not hr_size
            H,W = round(h*scale), round(w*scale)
            hr_size = (H,W)

        if scale not in self.cost_list:
            h0,w0 = 16,16
            H0,W0 = round(h0*scale), round(w0*scale)
            inp_coord = make_coord((H0,W0), flatten=True, device='cuda').unsqueeze(0)
            inp_cell = torch.ones_like(inp_coord)
            inp_cell[:,:,0] *= 2/H0
            inp_cell[:,:,1] *= 2/W0
            inp_encoder = torch.zeros((1,3,h0,w0), device='cuda')
            flops_encoder = get_model_flops(self.encoder, inp_encoder)
            inp_sampler = torch.zeros((1,self.encoder.out_dim,h0,w0), device='cuda')
            x = get_model_flops(self.light_sampler, inp_sampler, coord=inp_coord, cell=inp_cell)
            y = get_model_flops(self.heavy_sampler, inp_sampler, coord=inp_coord, cell=inp_cell)
            cost_list = torch.FloatTensor([x,y]).cuda() + flops_encoder
            cost_list = cost_list / cost_list.sum()
            self.cost_list[scale] = cost_list
            print('cost_list calculated (x{}): {}'.format(scale, cost_list))
        cost_list = self.cost_list[scale]

        feat = self.encoder(lr)
        b,q = coord.shape[:2]
        assert H*W == q
        tot = b*q
        if not pixel_batch_size: 
            pixel_batch_size = q

        # pre-calculate flag
        prob = torch.zeros((b,q,2), device=lr.device)
        pb = pixel_batch_size//b*b
        assert pb > 0
        l = 0
        while l < q:
            r = min(q, l+pb)
            coord_split = coord[:,l:r,:]
            cell_split = cell[:,l:r,:]
            prob_split = self.classifier(feat, coord_split, cell_split)
            prob[:,l:r] = F.softmax(prob_split, dim=-1)
            l = r      

        if adaptive_cluster: # auto-decide threshold
            diff = prob[:,:,1].view(-1,1) # (tot,1)
            assert diff.max() > diff.min()
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            centroids = torch.FloatTensor([[0.5]]).cuda()
            flag = self.kmeans.fit_predict(diff, centroids=centroids)
            _, min_index = torch.min(diff.flatten(), dim=0)
            if flag[min_index] == 1:
                flag = 1 - flag # (tot,)
            flag = flag.view(b,q)
        else:
            prob = prob / torch.pow(cost_list, k).view(1,1,2)
            flag = torch.argmax(prob, dim=-1) # (b,q)

        # inference per image
        # more efficient implementation may exist
        preds = []
        for i in range(b):
            pred = torch.zeros((q,3), device=lr.device)
            l = 0
            while l < q:
                r = min(q, l+pixel_batch_size)
                coord_split = coord[i:i+1,l:r,:]
                cell_split = cell[i:i+1,l:r,:]
                flg = flag[i,l:r]

                idx_easy = torch.where(flg == 0)[0]
                idx_hard = torch.where(flg == 1)[0]
                num_easy, num_hard = len(idx_easy), len(idx_hard)
                if num_easy > 0: 
                    pred[l+idx_easy] = self.light_sampler(feat[i:i+1], coord_split[:,idx_easy,:], cell_split[:,idx_easy,:]).squeeze(0)
                if num_hard > 0: 
                    pred[l+idx_hard] = self.heavy_sampler(feat[i:i+1], coord_split[:,idx_hard,:], cell_split[:,idx_hard,:]).squeeze(0)
                res = F.grid_sample(lr[i:i+1], coord_split.flip(-1).unsqueeze(1), mode='bilinear', 
                    padding_mode='border', align_corners=False)[:,:,0,:].squeeze(0).transpose(0,1)
                pred[l:r] += res
                l = r
            preds.append(pred)
        pred = torch.stack(preds, dim=0) # (b,q,3)

        if refinement:
            pred = pred.transpose(1,2).view(-1,3,H,W)
            pred_unfold = F.pad(pred, (1,1,1,1), mode='replicate')
            pred_unfold = F.unfold(pred_unfold, 3, padding=0).view(-1,3,9,H,W).mean(dim=2) # (b,3,H,W)
            flag = flag.view(-1,1,H,W)
            flag_unfold = F.pad(flag.float(), (1,1,1,1), mode='replicate')
            flag_unfold = F.unfold(flag_unfold, 3, padding=0).view(-1,1,9,H,W).int().sum(dim=2) # (b,1,H,W)

            cond = (flag==0) & (flag_unfold>0) #
            cond[:,:,[0,-1],:] = cond[:,:,:,[0,-1]] = False
            #print('refined: {} / {}'.format(cond.sum().item(), tot))
            pred = torch.where(cond, pred_unfold, pred)
            pred = pred.view(-1,3,q).transpose(1,2)
        flag = flag.view(b,q,1)
        return pred, flag