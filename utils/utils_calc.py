import numpy as np
import torch
from utils.utils_image import tensor2numpy

# https://github.com/cszn/KAIR
def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input: (h,w,3) np array
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def psnr_measure(src, tar, y_channel=False, shave_border=0):
    # np array must be 0-255, (h,w,3)
    # tensor must be 0-1, (3,h,w)
    if isinstance(src, torch.Tensor):
        assert isinstance(tar, torch.Tensor)
        if src.ndim == 4:
            src = src.squeeze(0)
        if tar.ndim == 4:
            tar = tar.squeeze(0)
        if y_channel:
            src = tensor2numpy(src)
            tar = tensor2numpy(tar)
            src = rgb2ycbcr(src).astype(np.float32, copy=False)
            tar = rgb2ycbcr(tar).astype(np.float32, copy=False)
        else:
            src = (src*255).clamp_(0,255).round().permute(1,2,0)
            tar = (tar*255).clamp_(0,255).round().permute(1,2,0)
    else:
        if y_channel:
            src = rgb2ycbcr(src)
            tar = rgb2ycbcr(tar)
        src = src.astype(np.float32, copy=False)
        tar = tar.astype(np.float32, copy=False) 
    diff = tar - src
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    
    if isinstance(diff, torch.Tensor):
        err = torch.mean(torch.pow(diff, 2)).item()
    else:
        err = np.mean(np.power(diff, 2))
    #if err < 0.6502:
    #    return 50
    #else:
    return 10 * np.log10((255. ** 2) / err)  
