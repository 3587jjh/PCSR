import argparse
import cv2
import os, glob, shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import yaml

import models
from flops import compute_num_params, get_model_flops
from utils import *
import core
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='original image path')
    parser.add_argument('--k', type=float, default=0.)
    parser.add_argument('--adaptive', action='store_true', help='whether to use ADM. k is not used')
    parser.add_argument('--no_refinement', action='store_false', help='whether not to use pixel-wise refinement')
    parser.add_argument('--opacity', type=float, default=0.65, help='opacity for colored visualization')
    parser.add_argument('--pixel_batch_size', type=int, default=300000)
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    # load original model
    scale = 4
    resume_path = 'save/carn-x4/iter_last.pth'
    sv_file = torch.load(resume_path)
    model_original = models.make(sv_file['model'], load_sd=True).cuda()
    model_original.eval()

    # load pcsr model
    resume_path = 'save/carn-pcsr-phase1/iter_last.pth'
    sv_file = torch.load(resume_path)
    model = models.make(sv_file['model'], load_sd=True).cuda()
    model.eval()

    rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor([1.0, 1.0, 1.0], device='cuda').view(1,3,1,1)
    out_dir = args.out_dir

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger()
    logger.set_save_path(out_dir, remove=False)
    log = logger.log

    log('backbone: CARN')
    log('model_original: #params={}'.format(compute_num_params(model_original, text=True)))
    log('model: #params={}'.format(compute_num_params(model, text=True)))

    with torch.no_grad():
        hr_path = args.img_path
        hr = transforms.ToTensor()(Image.open(hr_path)).unsqueeze(0) # (1,3,H,W), range=[0,1]

        # center-crop hr to scale-divisible size
        H,W = hr.shape[-2:]
        H,W = H//scale*scale, W//scale*scale
        h,w = H//scale, W//scale
        hr = center_crop(hr, (H,W)).cuda()
        lr = core.imresize(hr, sizes=(h,w)) # [0,1]
        inp_lr = (lr - rgb_mean) / rgb_std
        
        log('')
        log('hr_path: {}'.format(hr_path))
        log('scale: {}'.format(scale))
        log('hr resolution (center_cropped):({}, {})'.format(hr.shape[-2], hr.shape[-1]))
        log('lr resolution: ({}, {})'.format(lr.shape[-2], lr.shape[-1]))
 
        # model_original prediction
        pred_original = model_original(inp_lr)
        total_flops_original = get_model_flops(model_original, inp_lr)
        pred_original = pred_original * rgb_std + rgb_mean

        # model prediction
        coord = make_coord((H,W), flatten=True, device='cuda').unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[:,:,0] *= 2/H
        cell[:,:,1] *= 2/W
        pred, flag = model(inp_lr, coord=coord, cell=cell, scale=scale, k=args.k, 
            pixel_batch_size=args.pixel_batch_size, adaptive_cluster=args.adaptive, refinement=not args.no_refinement)
        total_flops = get_model_flops(model, inp_lr, coord=coord, cell=cell, scale=scale, k=args.k, 
            pixel_batch_size=args.pixel_batch_size, adaptive_cluster=args.adaptive, refinement=not args.no_refinement)
        pred = pred.transpose(1,2).view(-1,3,H,W)
        pred = pred * rgb_std + rgb_mean
        flag = flag.view(-1,1,H,W).repeat(1,3,1,1)

    # image-level evaluation
    psnr_original = psnr_measure(pred_original, hr, y_channel=False, shave_border=scale)
    psnr = psnr_measure(pred, hr, y_channel=False, shave_border=scale)
    log('Original: {:.2f} dB | {:.2f}G (100 %)'.format(psnr_original, total_flops_original/1e9))
    log('Ours: {:.2f} dB | {:.2f}G ({} %)'.format(psnr, total_flops/1e9, round(total_flops/total_flops_original*100)))

    # visualization
    hr = tensor2numpy(hr)
    lr = tensor2numpy(lr)
    pred_original = tensor2numpy(pred_original)
    pred = tensor2numpy(pred)

    Image.fromarray(hr).save(f'{out_dir}/HR.png')
    Image.fromarray(lr).save(f'{out_dir}/LRx4.png')
    Image.fromarray(pred_original).save(f'{out_dir}/Original.png')
    Image.fromarray(pred).save(f'{out_dir}/PCSR.png')

    flag = flag.squeeze(0).detach().cpu()
    H,W = hr.shape[:2]
    red = np.array([255,0,0])
    green = np.array([0,255,0])
    opacity = args.opacity

    vis_img = np.zeros_like(hr)
    vis_img[flag[0] == 0] = green
    vis_img[flag[0] == 1] = red
    vis_img = vis_img*(1-opacity) + pred*opacity
    Image.fromarray(vis_img.astype('uint8')).save(f'{out_dir}/PCSR_colored.png')


if __name__ == '__main__':
    main()