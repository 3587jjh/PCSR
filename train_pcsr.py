import os
import argparse
import yaml
import builtins

from utils import *
from flops import compute_num_params, get_model_flops

import datasets
import models
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore")


def prepare_training(config, log):
    resume_path = config['resume_path']
    resume = os.path.exists(resume_path)

    if resume:
        sv_file = torch.load(resume_path, map_location=config['map_loc'])
        iter_start = sv_file['iter']+1
        if iter_start <= config['iter_max']//100:
            resume = False
        else:
            log('Model resumed from: {} (prev_iter: {})'.format(resume_path, sv_file['iter']))
            model = models.make(sv_file['model'], load_sd=True).cuda()
            optimizer, lr_scheduler = make_optim_sched(model.parameters(),
                sv_file['optimizer'], sv_file['lr_scheduler'], load_sd=True)

    if not resume:
        assert not config.get('init_path')
        if config['phase'] == 0:
            log('Loading new model ...')
            model = models.make(config['model']).cuda()
        else:
            model = models.make(config['model']).cuda()
            save_path = config['save_path'][:-1] + '0' # previous phase
            config['init_path'] = '{}/iter_last.pth'.format(save_path)

            sv_file = torch.load(config['init_path'], map_location=config['map_loc'])
            init_model = models.make(sv_file['model'], load_sd=True).cuda()
            log('[encoder] [heavy sampler] init from ... {}'.format(config['init_path']))
            model.encoder = init_model.encoder
            model.heavy_sampler = init_model.heavy_sampler
        optimizer, lr_scheduler = make_optim_sched(model.parameters(),
            config['optimizer'], config['lr_scheduler'])
        iter_start = 1

    for param in model.parameters():
        param.requires_grad = True
    if config['phase'] == 1:
        model.encoder.requires_grad_(False)
        model.heavy_sampler.requires_grad_(False)
        log('freeze: [encoder] [heavy sampler]')

    if config['rank'] == 0:
        psz = config['patch_size']
        x = torch.zeros((1,3,psz,psz), device='cuda')
        model.eval()
        log('patch_size: {}'.format(psz))
        for scale in config['valid_dataset']['scales']:
            L = psz * scale
            coord = make_coord((L,L), flatten=True, device='cuda').unsqueeze(0)
            cell = torch.ones_like(coord)
            cell[:,:,0] *= 2/L
            cell[:,:,1] *= 2/L
            flops_encoder = get_model_flops(model.encoder, x)

            if config['phase'] == 0:
                flops_heavy = get_model_flops(model, x, coord=coord, cell=cell)
                log('scale: x{} | encoder flops: {:.0f}M ({:.0f}%) | heavy flops: {:.0f}M (100%)'\
                    .format(scale, flops_encoder/1e6, flops_encoder/flops_heavy*100, flops_heavy/1e6))
            else:
                feat = torch.zeros((1, model.encoder.out_dim, psz, psz), device='cuda')
                flops_heavy = get_model_flops(model.heavy_sampler, feat, coord=coord, cell=cell) + flops_encoder
                flops_light = get_model_flops(model.light_sampler, feat, coord=coord, cell=cell) + flops_encoder
                log('scale: x{} | light flops: {:.0f}M | heavy flops: {:.0f}M'\
                    .format(scale, flops_light/1e6, flops_heavy/1e6))
    log('#params={}'.format(compute_num_params(model, text=True)))
    #exit()
    return model, optimizer, lr_scheduler, iter_start


def make_data_loader(config, tag, eval_scale=None):
    spec = config[f'{tag}_dataset']
    dataset = datasets.make(spec['dataset'])

    if tag == 'train':
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
        assert spec['batch_size'] % config['world_size'] == 0
        batch_size = spec['batch_size'] // config['world_size']
        assert spec['num_workers'] % config['world_size'] == 0
        num_workers = spec['num_workers'] // config['world_size']
        drop_last = True
        seed = 0 if not config['seed'] else config['seed']
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    else: # valid
        assert eval_scale
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'scale': eval_scale})
        batch_size = 1
        num_workers = 1
        drop_last = False
        sampler = None
    
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
        shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return data_loader, sampler


def valid(model, config, scale, pixel_batch_size=800000, k=0.):
    model.eval()
    valid_loader, _ = make_data_loader(config, 'valid', scale)
    psnrs = []
    total_flops = 0
    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    for batch in tqdm(valid_loader, leave=True, desc=f'valid (x{scale})'):
        for key, value in batch.items():
            batch[key] = value.cuda()

        lr = (batch['lr'] - rgb_mean) / rgb_std
        hr = batch['hr']
        H,W = hr.shape[-2:]

        with torch.no_grad():
            if config['phase'] == 0:
                pred = model(lr, batch['coord'], batch['cell'], 
                    pixel_batch_size=pixel_batch_size)
                total_flops += get_model_flops(model, lr, coord=batch['coord'], cell=batch['cell'], 
                    pixel_batch_size=pixel_batch_size)
            else:
                pred, _ = model(lr, batch['coord'], batch['cell'], scale=scale, 
                    k=k, pixel_batch_size=pixel_batch_size, refinement=False)
                total_flops += get_model_flops(model, lr, coord=batch['coord'], cell=batch['cell'], scale=scale, 
                    k=k, pixel_batch_size=pixel_batch_size, refinement=False)
                    
        pred = pred.transpose(1,2).view(-1,3,H,W)
        pred = pred * rgb_std + rgb_mean
        psnr = psnr_measure(pred, hr, y_channel=(config['psnr_type'] != 'rgb'), shave_border=scale) 
        psnrs.append(psnr)

    psnr = np.mean(np.array(psnrs))
    avg_flops = total_flops / len(valid_loader)
    return psnr, avg_flops


def main(): 
    # get options
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # distributed setting
    init_dist('pytorch')
    rank, world_size = get_dist_info()

    # load logger
    save_path = os.path.join('save', args.config.split('/')[-1][:-len('.yaml')])
    logger = Logger()
    logger.set_save_path(save_path, remove=False)
    if rank > 0: 
        builtins.print = lambda *args, **kwargs: None
        logger.disable()
    log = logger.log

    # load config
    config = load_config(args.config)
    config['world_size'] = world_size
    if config['seed'] is not None:
        set_seed(config['seed'])
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    log('Config loaded: {}'.format(args.config))

    config['rank'] = rank
    config['map_loc'] = f'cuda:{rank}'
    phase = config['phase']
    if rank == 0:
        assert (phase == 0 or phase == 1)

    # prepare training
    model, optimizer, lr_scheduler, iter_start = prepare_training(config, log)
    model = nn.parallel.DistributedDataParallel(model)
    train_loader, train_sampler = make_data_loader(config, 'train')
    if rank == 0:
        timer = Timer()
        train_loss = Averager()
        t_iter_start = timer.t()

    if phase == 0:
        loss_fn = nn.L1Loss()
    else:
        loss_fn_rgb = nn.L1Loss()
        loss_fn_avg = nn.L1Loss()
        if rank == 0:
            train_loss_rgb = Averager()
            train_loss_avg = Averager()

    iter_cur = iter_start
    iter_max = config['iter_max']
    iter_print = config['iter_print']
    iter_val = config['iter_val']
    iter_save = config['iter_save']

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda')
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda')

    while True:
        train_sampler.set_epoch(iter_cur) # instead of epoch
        for batch in train_loader:
            # process single iteration
            model.train()
            optimizer.zero_grad()
            if phase == 1:
                model.module.encoder.eval()
                model.module.heavy_sampler.eval()

            for key, value in batch.items():
                batch[key] = value.cuda()

            lr = (batch['lr'] - rgb_mean.view(1,3,1,1)) / rgb_std.view(1,3,1,1)
            hr_rgb = (batch['hr_rgb'] - rgb_mean.view(1,1,3)) / rgb_std.view(1,1,3)

            if phase == 0:
                pred_heavy = model(lr, batch['coord'], batch['cell'])
                loss = loss_fn(pred_heavy, hr_rgb)
            else:
                pred, prob = model(lr, batch['coord'], batch['cell']) # (b,q,3), (b,q,2)
                target_cnt = torch.ones(1, device='cuda') * prob.shape[0] * prob.shape[1] / 2
                loss_rgb = loss_fn_rgb(pred, hr_rgb)
                loss_avg = loss_fn_avg(prob[:,:,1].sum(), target_cnt) / target_cnt
                loss = loss_rgb * config['loss_rgb_w'] + loss_avg * config['loss_avg_w']

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                train_loss.add(loss.item())
                if phase == 1:
                    train_loss_rgb.add(loss_rgb.item() * config['loss_rgb_w'])
                    train_loss_avg.add(loss_avg.item() * config['loss_avg_w'])

                cond1 = (iter_cur % iter_print == 0)
                cond2 = (iter_cur % iter_save == 0)
                cond3 = (iter_cur % iter_val == 0)

                if cond1 or cond2 or cond3:
                    model_ = model.module
                    if cond1 or cond2:
                        # save current model state
                        model_spec = config['model']
                        model_spec['sd'] = model_.state_dict()
                        optimizer_spec = config['optimizer']
                        optimizer_spec['sd'] = optimizer.state_dict()
                        lr_scheduler_spec = config['lr_scheduler']
                        lr_scheduler_spec['sd'] = lr_scheduler.state_dict()
                        sv_file = {
                            'model': model_spec,
                            'optimizer': optimizer_spec,
                            'lr_scheduler': lr_scheduler_spec,
                            'iter': iter_cur
                        }
                        if cond1:
                            log_info = ['iter {}/{}'.format(iter_cur, iter_max)]
                            if phase == 0:
                                log_info.append('train: loss={:.4f}'.format(train_loss.item()))
                            else:
                                log_info.append('train: loss={:.4f} | loss_rgb={:.4f} | loss_avg={:.4f}'\
                                    .format(train_loss.item(), train_loss_rgb.item(), train_loss_avg.item()))
                            log_info.append('lr: {:.4e}'.format(lr_scheduler.get_last_lr()[0]))

                            t = timer.t()
                            prog = (iter_cur - iter_start + 1) / (iter_max - iter_start + 1)
                            t_iter = time_text(t - t_iter_start)
                            t_elapsed, t_all = time_text(t), time_text(t / prog)
                            log_info.append('{} {}/{}'.format(t_iter, t_elapsed, t_all))
                            log(', '.join(log_info))

                            train_loss = Averager()
                            if phase == 1:
                                train_loss_rgb = Averager()
                                train_loss_avg = Averager()
                            t_iter_start = timer.t()
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_last.pth'))

                        if cond2:
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_{}.pth'.format(iter_cur)))
                    if cond3: # validation
                        for scale in config['valid_dataset']['scales']:
                            if phase == 0:
                                psnr, flops = valid(model_, config, scale)
                                log('valid (x{}) | psnr({}): {:.2f} dB | flops (per image): {:.2f}G'\
                                    .format(scale, config['psnr_type'], psnr, flops/1e9))
                            else:
                                psnr_heavy, flops_heavy = valid(model_, config, scale, k=-25)
                                psnr, flops = valid(model_, config, scale, k=0)
                                log('valid (x{}) | psnr_mix({}): {:.2f} dB | psnr_heavy({}): {:.2f} dB | flops_ratio: {:.1f} %'\
                                    .format(scale, config['psnr_type'], psnr, config['psnr_type'], 
                                        psnr_heavy, flops/flops_heavy*100))
            if iter_cur == iter_max:
                log('Finish training.')
                return
            iter_cur += 1

if __name__ == '__main__':
    main()