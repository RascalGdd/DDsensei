import torch

from paired import MatchedCrops
from crop import *
from cfg import *
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as tf

def get_dataloader():

    dataset_fake = ImageDataset(file_list_fake_label,for_label=True)
    dataset_real = ImageDataset(file_list_real)
    dataset_fake2 = ImageDataset(file_list_fake)
    data = MatchedCrops(dataset_fake, dataset_real,dataset_fake2, matched_crop_path, weight_path)

    loader = DataLoader(data, batch_size=2, shuffle=True)
    for i in loader:
        k = i[0]
        j = i[1]
        m = i[2]
        break
    # input_label = torch.FloatTensor(2, 35, 256, 256).zero_()
    # input_semantics = input_label.scatter_(1, k, 1.0)
    print("cropdataset prepared!")
    print("input label size: {}".format(k.shape))
    print("input image size: {}".format(j.shape))
    print("input image2 size: {}".format(m.shape))
    return loader

