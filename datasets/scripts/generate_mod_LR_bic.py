import os
import sys
sys.path.append('../..')

import cv2
import numpy as np
from tqdm import tqdm
from datasets.scripts.util import imresize_np


def generate_mod_LR_bic():
    up_scale = 4 ###
    mod_scale = 4 ###
    sourcedir = '/workspace/datasets/train/DIV2K/HR_aug_sub128' ### set path
    savedir = '/workspace/datasets/train/DIV2K/Aug_sub' ### set path
    saveHRpath = os.path.join(savedir, 'HR', 'X' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR_bicubic', 'X' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit()
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR_bicubic')):
        os.mkdir(os.path.join(savedir, 'LR_bicubic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png') or f.endswith('.jpg')]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in tqdm(range(num_files)):
        filename = filepaths[i]
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop (left-corner crop)
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)


if __name__ == "__main__":
    generate_mod_LR_bic()
