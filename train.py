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
        if config.get('init_path'):
            log('Model init from: {}'.format(config['init_path']))
            sv_file = torch.load(config['init_path'], map_location=config['map_loc'])    
            model = models.make(sv_file['model'], load_sd=True).cuda()
        else:
            log('Loading new model ...')
            model = models.make(config['model']).cuda()
        optimizer, lr_scheduler = make_optim_sched(model.parameters(),
            config['optimizer'], config['lr_scheduler'])
        iter_start = 1

    log('#params={}'.format(compute_num_params(model, text=True)))
    if config['rank'] == 0:
        psz = config['patch_size']
        x = torch.zeros((1,3,psz,psz), device='cuda')
        flops_model = get_model_flops(model, x)
        log('model flops ({}x{}): {:.0f}M'.format(psz, psz, flops_model/1e6))
    #exit()
    return model, optimizer, lr_scheduler, iter_start


def make_data_loader(config, tag):
    spec = config[f'{tag}_dataset']
    seed = 0 if not config['seed'] else config['seed']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if tag == 'train':
        assert spec['batch_size'] % config['world_size'] == 0
        batch_size = spec['batch_size'] // config['world_size']
        assert spec['num_workers'] % config['world_size'] == 0
        num_workers = spec['num_workers'] // config['world_size']
        drop_last = True
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    else:
        batch_size = 1
        num_workers = 1
        drop_last = False
        sampler = None
    
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
        shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return data_loader, sampler


def valid(model, config, valid_loader):
    model.eval()
    psnrs = []
    scale = config['scale']
    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    for i, batch in enumerate(tqdm(valid_loader, leave=True, desc=f'valid (x{scale})')):
        for key, value in batch.items():
            batch[key] = value.cuda()

        lr = (batch['lr'] - rgb_mean) / rgb_std
        hr = batch['hr']

        with torch.no_grad():
            pred = model(lr) # (b,c,h,w)
            pred = pred * rgb_std + rgb_mean

        psnr = psnr_measure(pred, hr, y_channel=(config['psnr_type'] != 'rgb'), shave_border=scale) 
        psnrs.append(psnr)
    psnr = np.mean(np.array(psnrs))
    return psnr


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

    # prepare training
    model, optimizer, lr_scheduler, iter_start = prepare_training(config, log)
    model = nn.parallel.DistributedDataParallel(model)
    loss_fn = nn.L1Loss()
    train_loader, train_sampler = make_data_loader(config, 'train')

    if rank == 0:
        valid_loader, _ = make_data_loader(config, 'valid')
        timer = Timer()
        train_loss = Averager()
        t_iter_start = timer.t()

    iter_cur = iter_start
    iter_max = config['iter_max']
    iter_print = config['iter_print']
    iter_val = config['iter_val']
    iter_save = config['iter_save']

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    while True:
        train_sampler.set_epoch(iter_cur) # instead of epoch
        for batch in train_loader:
            # process single iteration
            model.train()
            optimizer.zero_grad()
            for key, value in batch.items():
                batch[key] = value.cuda()

            lr = (batch['lr'] - rgb_mean) / rgb_std
            hr = (batch['hr'] - rgb_mean) / rgb_std

            pred = model(lr)
            loss = loss_fn(pred, hr)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                train_loss.add(loss.item())
                cond1 = (iter_cur % iter_print == 0)
                cond2 = (iter_cur % iter_save == 0)
                cond3 = (iter_cur % iter_val == 0)

                if cond1 or cond2 or cond3:
                    model_ = model.module if hasattr(model, 'module') else model
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
                            log_info.append('train: loss={:.4f}'.format(train_loss.item()))
                            log_info.append('lr: {:.4e}'.format(lr_scheduler.get_last_lr()[0]))

                            t = timer.t()
                            prog = (iter_cur - iter_start + 1) / (iter_max - iter_start + 1)
                            t_iter = time_text(t - t_iter_start)
                            t_elapsed, t_all = time_text(t), time_text(t / prog)
                            log_info.append('{} {}/{}'.format(t_iter, t_elapsed, t_all))
                            log(', '.join(log_info))
                            train_loss = Averager()
                            t_iter_start = timer.t()
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_last.pth'))
                        if cond2:
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_{}.pth'.format(iter_cur)))
                    if cond3: # validation
                        psnr = valid(model_, config, valid_loader)
                        log('valid (x{}) | psnr({}): {:.2f} dB'.format(config['scale'], config['psnr_type'], psnr))

            if iter_cur == iter_max:
                log('Finish training.')
                return
            iter_cur += 1

if __name__ == '__main__':
    main()