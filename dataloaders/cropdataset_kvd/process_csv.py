import pandas as pd
import os
import PIL.Image as Image
import numpy as np
import cv2
from dataloaders.cropdataset_kvd.cfg2 import *

def crop(img, r0, r1, c0, c1):
    """ Return cropped patch from image tensor(s). """

    return img[r0:r1, c0:c1]


data = pd.read_csv(matched_crop_path)
droplist = []
for i in range(data.shape[0]):
    path1, r0, r1, c0, c1 = data.iloc[i][0:5]
    lbl1 = Image.open(path1).resize((752, 408), resample=Image.NEAREST)
    lbl1 = np.array(lbl1)
    lbl1 = crop(lbl1, r0, r1, c0, c1)
    lbl1 = cv2.resize(lbl1, [16, 16], interpolation=cv2.INTER_NEAREST)

    path2, r0, r1, c0, c1 = data.iloc[i][5:]
    lbl2 = Image.open(path2).resize((752, 408), resample=Image.NEAREST)
    lbl2 = np.array(lbl2)
    lbl2 = crop(lbl2, r0, r1, c0, c1)
    lbl2 = cv2.resize(lbl2, [16, 16], interpolation=cv2.INTER_NEAREST)

    score = np.sum(lbl1==lbl2)
    total = 256.
    percent = score / total
    if percent < 0.5:
        droplist.append(i)


data.drop(droplist, inplace=True)
data.reset_index(drop=True, inplace=True)
data.to_csv(matched_crop_path_kvd)

