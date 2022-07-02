import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as tf
from torchvision.utils import save_image
import cv2

file_path = os.path.join("D:\labels")
test_path = os.path.join("D:\labels//00001.png")
gtav_dataroot = '/data/public/gta/labels'
my_dataroot = '/no_backups/s1422/patchdata'
local_root = "C://Users//guodi//Desktop//save"
cityscape_root = '/data/public/cityscapes/leftImg8bit/train'

# img = cv2.imread(test_path)

# folder_list = os.listdir(cityscape_root)
file_list = os.listdir(gtav_dataroot)
idx = 1
# for folder in folder_list:
#     print(folder)
#     new_path = os.path.join(cityscape_root,folder)
#     file_list = os.listdir(new_path)
for i in file_list:
    img_path = os.path.join(file_path,i)
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    img = np.array(img)
    h = img.shape[0]
    w = img.shape[1]
    a = int(h/4)
    b = int(w/4)
    # print(np.unique(img))
    # img = cv2.resize(img, dsize=[512, 256])
    for k in range(4):
        for m in range(4):
            patch = img[k*a:(k*a+a), m*b:(m*b+b)]
            patch = cv2.resize(patch, dsize=[512, 256],interpolation=cv2.INTER_NEAREST)
            # with np.printoptions(threshold=np.inf):
            #     print(np.unique(patch))
            # print(patch)
            if idx >= 1 and idx<10:
                name = "0000" + str(idx) + ".png"
            elif idx >= 10 and idx<100:
                name = "000" + str(idx) + ".png"
            elif idx >= 100 and idx<1001:
                name = "00" + str(idx) + ".png"
            elif idx >= 1000 and idx<10000:
                name = "0" + str(idx) + ".png"
            else:
                name = str(idx) + ".png"
            idx += 1
            path = os.path.join(my_dataroot, name)
            cv2.imwrite(path,patch)
