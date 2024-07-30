import numpy as np
from torch.utils.data import Dataset
from datasets import register
from utils import *
import core


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=None):
        self.inp_size = inp_size
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr, lr = self.dataset[idx] # (3,h,w) tensor, range [0,1]
        if self.inp_size:
            hr, lr = random_crop_together(hr, lr, self.inp_size)

        if self.augment: # when training
            hflip = (np.random.random() < 0.5) if 'hflip' in self.augment else False
            vflip = (np.random.random() < 0.5) if 'vflip' in self.augment else False
            dflip = (np.random.random() < 0.5) if 'dflip' in self.augment else False

            def base_augment(img):
                if hflip:
                    img = img.flip(-2)
                if vflip:
                    img = img.flip(-1)
                if dflip:
                    img = img.transpose(-2, -1)
                return img
            hr = base_augment(hr)
            lr = base_augment(lr)

        return {
            'lr': lr,
            'hr': hr
        }

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale=[1,4], sample_q=None, augment=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale = scale
        self.sample_q = sample_q
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def getitem_train(self, hr):
        smin, smax = self.scale
        s = np.random.uniform(smin, smax)
        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        hr = random_crop(hr, w_hr)

        # augmentation
        hflip = (np.random.random() < 0.5) if 'hflip' in self.augment else False
        vflip = (np.random.random() < 0.5) if 'vflip' in self.augment else False
        dflip = (np.random.random() < 0.5) if 'dflip' in self.augment else False

        def base_augment(img):
            if hflip:
                img = img.flip(-2)
            if vflip:
                img = img.flip(-1)
            if dflip:
                img = img.transpose(-2, -1)
            return img
        hr = base_augment(hr)
        lr = core.imresize(hr, sizes=(w_lr, w_lr))

        hr_coord, hr_rgb = to_pixel_samples(hr.contiguous())
        # sample pixels
        sample_q = self.sample_q if self.sample_q else (self.inp_size*smin)**2
        sample_lst = np.random.choice(len(hr_coord), sample_q, replace=False)
        hr_coord = hr_coord[sample_lst]
        hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2/hr.shape[-2]
        cell[:, 1] *= 2/hr.shape[-1]

        return {
            'lr': lr,
            'coord': hr_coord,
            'cell': cell,
            'hr_rgb': hr_rgb
        }

    def getitem_eval(self, hr):
        assert isinstance(self.scale, int)
        assert not self.augment and not self.sample_q
        s = self.scale
        H,W = hr.shape[-2:]
        if H%s != 0 or W%s != 0:
            print('(eval) image cropped to scale-divisible size')
            H,W = H//s*s, W//s*s
            hr = hr[:,:H,:W]
        h,w = H//s, W//s
        lr = core.imresize(hr, sizes=(h,w))
        
        hr_coord, hr_rgb = to_pixel_samples(hr.contiguous())
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2/hr.shape[-2]
        cell[:, 1] *= 2/hr.shape[-1]

        return {
            'lr': lr,
            'coord': hr_coord,
            'cell': cell,
            'hr_rgb': hr_rgb,
            'hr': hr
        }

    def __getitem__(self, idx):
        hr = self.dataset[idx] # (3,H,W) tensor, range [0,1]
        return self.getitem_train(hr) if self.inp_size else self.getitem_eval(hr)
