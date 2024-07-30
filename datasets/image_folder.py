import os
from PIL import Image
import pickle
import imageio

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register
from utils.utils_io import *


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, first_k=None, repeat=1, cache='none'):
        self.repeat = repeat
        if root_path.endswith('.lmdb'):
            self.is_lmdb = True
            img_list = sorted(list(scandir(root_path.split('.')[0], recursive=False)))
            self.files = [img_path.split('.')[0] for img_path in sorted(img_list)] # keys
            if first_k is not None:
                self.files = self.files[:first_k]
            self.client_key = 'hr' if 'HR' in root_path else 'lr'
            self.io_backend_opt = {
                'type': 'lmdb',
                'db_paths': root_path,
                'client_keys': self.client_key
            }
            self.file_client = None
        else:
            self.cache = cache
            self.is_lmdb = False
            filenames = sorted(os.listdir(root_path))
            if first_k is not None:
                filenames = filenames[:first_k]
                
            self.files = []
            for filename in filenames:
                file = os.path.join(root_path, filename)
                if cache == 'none':
                    self.files.append(file) # img_path only, when data is too big
                elif cache == 'in_memory': # no more I/O, when data is small enough
                    self.files.append(transforms.ToTensor()(
                        Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.is_lmdb:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            x = self.file_client.get(x, self.client_key)
            x = imfrombytes(x, float32=True)
            x = img2tensor(x)
        else:
            if self.cache == 'none':
                x = Image.open(x).convert('RGB')
                x = transforms.ToTensor()(x)
        return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_hr, root_path_lr, **kwargs):
        self.dataset_hr = ImageFolder(root_path_hr, **kwargs)
        self.dataset_lr = ImageFolder(root_path_lr, **kwargs)

    def __len__(self):
        return len(self.dataset_lr)

    def __getitem__(self, idx):
        return self.dataset_hr[idx], self.dataset_lr[idx]