import os
import sys
sys.path.append('../..')

import cv2
from tqdm import tqdm
from datasets.scripts.util import _get_paths_from_images, imresize_np

#we first downsample the original images with scaling factors 0.6, 0.7, 0.8, 0.9 to generate the HR/LR images.
for scale in [1, 0.9, 0.8, 0.7, 0.6]:
    GT_folder = '/workspace/datasets/train/DIV2K/HR' ### set path
    save_GT_folder = '/workspace/datasets/train/DIV2K/HR_aug' ### set path
    for i in [save_GT_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    img_GT_list = _get_paths_from_images(GT_folder)
    for path_GT in tqdm(img_GT_list):
        img_GT = cv2.imread(path_GT)
        img_GT = img_GT
        # imresize
        rlt_GT = imresize_np(img_GT, scale, antialiasing=True)
        #print(str(scale) + "_" + os.path.basename(path_GT))
        
        if scale == 1:
            cv2.imwrite(os.path.join(save_GT_folder,os.path.basename(path_GT)), rlt_GT)
        else:
            cv2.imwrite(os.path.join(save_GT_folder, str(scale) + "_" + os.path.basename(path_GT)), rlt_GT)