import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets
import models
from utils import *
from flops import compute_num_params, get_model_flops
from tqdm import tqdm
from time import time

import warnings
warnings.filterwarnings("ignore")


def load_model():
    resume_path = config['resume_path']
    print('Model resumed from ...', resume_path)
    sv_file = torch.load(resume_path)
    model = models.make(sv_file['model'], load_sd=True).cuda()
    print('model: #params={}'.format(compute_num_params(model, text=True)))
    return model


def make_test_loader(): 
    spec = config['test_dataset']
    spec['dataset']['args']['root_path'] = config['hr_data']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'scale': scale})
    test_loader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=1, pin_memory=True)
    return test_loader


def test(model):
    model.eval()
    test_loader = make_test_loader()
    psnrs = []
    total_flops = 0
    total_patches = 0

    crop_sz = config['patch_size']
    step = config['step']
    patch_batch_size = config['patch_batch_size']
    pixel_batch_size = config['pixel_batch_size']

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    if not config['per_image']:
        L = crop_sz * scale
        coord_patch = make_coord((L,L), flatten=True, device='cuda').unsqueeze(0)
        cell_patch = torch.ones_like(coord_patch)
        cell_patch[:,:,0] *= 2/L
        cell_patch[:,:,1] *= 2/L

    total_time = 0
    for batch in tqdm(test_loader, leave=True, desc=f'test (x{scale})'):
        for key, value in batch.items():
            batch[key] = value.cuda()
        
        lr = (batch['lr'] - rgb_mean) / rgb_std
        hr = batch['hr']

        h,w = lr.shape[-2:]
        num_patches = ((h-crop_sz+step)//step) * ((w-crop_sz+step)//step)
        total_patches += num_patches

        with torch.no_grad():
            if config['per_image']:
                if config['crop']:
                    # left corner crop lr, hr to patch-divisible size (for fair comparison)
                    crop_h = ((h-crop_sz+step)//step-1)*step+crop_sz
                    crop_w = ((w-crop_sz+step)//step-1)*step+crop_sz
                    lr = lr[:,:, :crop_h, :crop_w]
                    hr = hr[:,:, :scale*crop_h, :scale*crop_w]
                # generate coord, cell
                start_time = time()
                H,W = hr.shape[-2:]
                coord = make_coord((H,W), flatten=True, device='cuda').unsqueeze(0)
                cell = torch.ones_like(coord)
                cell[:,:,0] *= 2/H
                cell[:,:,1] *= 2/W
                end_time = time()
                total_time += end_time - start_time

                if phase == 0:
                    pred = model(lr, coord, cell, pixel_batch_size=pixel_batch_size)
                    total_flops += get_model_flops(model, lr, coord=coord, cell=cell,
                        pixel_batch_size=pixel_batch_size)
                else:
                    start_time = time()
                    pred, _, = model(lr, coord, cell, scale=scale, k=config['k'], 
                        pixel_batch_size=pixel_batch_size, adaptive_cluster=config['adaptive'])
                    end_time = time()
                    total_flops += get_model_flops(model, lr, coord=coord, cell=cell, scale=scale, k=config['k'], 
                        pixel_batch_size=pixel_batch_size, adaptive_cluster=config['adaptive'])
                    total_time += end_time - start_time
                pred = pred.transpose(1,2).view(-1,3,H,W)
                pred = pred * rgb_std + rgb_mean

            else:
                # left-corner crop lr, hr to patch-divisible size
                assert config['crop']
                crop_h = ((h-crop_sz+step)//step-1)*step+crop_sz
                crop_w = ((w-crop_sz+step)//step-1)*step+crop_sz
                lr = lr[:,:, :crop_h, :crop_w]
                hr = hr[:,:, :scale*crop_h, :scale*crop_w]

                # extract patches (w/o padding)
                lrs = nn.Unfold(kernel_size=crop_sz, stride=step)(lr)
                lrs = lrs.transpose(0,2).contiguous().view(-1,3,crop_sz,crop_sz)
                L = crop_sz*scale

                # batched(patch) model prediction
                preds = []
                l = 0
                while l < num_patches:
                    r = min(num_patches, l+patch_batch_size)
                    if phase == 0:
                        pred = model(lrs[l:r], coord_patch.repeat(r-l,1,1), cell_patch.repeat(r-l,1,1))
                        total_flops += get_model_flops(model, lrs[l:r],
                            coord=coord_patch.repeat(r-l,1,1),
                            cell=cell_patch.repeat(r-l,1,1))
                    else:
                        pred, _, = model(lrs[l:r], coord_patch.repeat(r-l,1,1), cell_patch.repeat(r-l,1,1), 
                            scale=scale, k=config['k'], adaptive_cluster=config['adaptive'])
                        total_flops += get_model_flops(model, lrs[l:r],
                            coord=coord_patch.repeat(r-l,1,1),
                            cell=cell_patch.repeat(r-l,1,1),
                            scale=scale, k=config['k'],
                            adaptive_cluster=config['adaptive'])

                    pred = pred.transpose(1,2).view(-1,3,L,L)
                    pred = pred * rgb_std + rgb_mean
                    preds.append(pred)
                    l = r
                preds = torch.cat(preds, dim=0)

                # combine preds
                preds = preds.flatten(1).unsqueeze(-1).transpose(0,2)
                mask = torch.ones_like(preds)
                mask = nn.Fold(output_size=hr.shape[-2:],
                    kernel_size=scale*crop_sz, stride=scale*step)(mask)
                pred = nn.Fold(output_size=hr.shape[-2:],
                    kernel_size=scale*crop_sz, stride=scale*step)(preds)/mask

            psnr = psnr_measure(pred, hr, y_channel=(config['psnr_type'] != 'rgb'), shave_border=scale) 
            psnrs.append(psnr)

    if not config['per_image']:      
        print('total_patches:', total_patches)
    psnr = np.mean(np.array(psnrs))
    avg_flops = total_flops / len(test_loader)
    print('total_time: {:.1f}s'.format(total_time))
    return psnr, avg_flops


def main(config_):
    global config, scale, phase
    config = config_
    scale = config['scale']
    phase = config['phase']

    model = load_model()
    psnr, flops = test(model)
    print('test (x{}) | psnr({}): {:.2f} dB | flops (per image): {:.2f}G'\
        .format(scale, config['psnr_type'], psnr, flops/1e9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--hr_data', type=str, required=True, help='hr data path')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--per_image', action='store_true', help='whether to per-image processing') # image bs=1
    parser.add_argument('--crop', action='store_true', help='whether to crop to patch-divisible size')
    parser.add_argument('--patch_batch_size', type=int, default=96)
    parser.add_argument('--pixel_batch_size', type=int, default=300000)
    parser.add_argument('--k', type=float, default=0.)
    parser.add_argument('--adaptive', action='store_true', help='whether to use ADM. k is not used')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = load_config(args.config)
    config['hr_data'] = args.hr_data
    config['scale'] = args.scale
    config['per_image'] = args.per_image
    config['crop'] = args.crop
    config['patch_batch_size'] = args.patch_batch_size
    config['pixel_batch_size'] = args.pixel_batch_size
    config['k'] = args.k
    config['adaptive'] = args.adaptive
    print('Config loaded ...', args.config)
    main(config)
