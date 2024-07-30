import os, sys
import shutil
import time

import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False # slower training
    cudnn.deterministic = True # slower training


class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        self.ignore = False

    def set_log_path(self, path):
        self.log_path = path

    def disable(self):
        self.ignore = True

    def log(self, obj, filename='log.txt'):
        if not self.ignore:
            print(obj)
            if self.log_path is not None:
                with open(os.path.join(self.log_path, filename), 'a') as f:
                    print(obj, file=f)

    @staticmethod
    def ensure_path(path, remove=True):
        basename = os.path.basename(path.rstrip('/'))
        if os.path.exists(path):
            if remove and (basename.startswith('_') or input('{} exists, remove? (y/[n]): '.format(path)).lower() == 'y'):
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)

    def set_save_path(self, save_path, remove=True):
        self.ensure_path(save_path, remove=remove)
        self.set_log_path(save_path)
        return self.log


def make_coord(shape, ranges=None, flatten=True, device='cpu'):
    # Make coordinates at grid centers.
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img, device='cpu'):
    """ 
    Convert the image to coord-RGB pairs.
    img: Tensor, (3, H, W)
    """
    if img.ndim == 3:
        coord = make_coord(img.shape[-2:], device=device)
        rgb = img.view(3, -1).permute(1, 0)
    else:
        b = img.shape[0]
        coord = make_coord(img.shape[-2:], device=device)
        rgb = img.view(b, 3, -1).permute(0, 2, 1)
    return coord, rgb


class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)