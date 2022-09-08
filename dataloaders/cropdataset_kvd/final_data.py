import torch
import numpy as np
from paired import MatchedCrops
from crop import *
from cfg import *
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as tf
import PIL.Image as Image
from torchvision.utils import save_image

if __name__ == '__main__':


    dataset_fake = ImageDataset(file_list_fake_label,for_label=True)
    dataset_fake2 = ImageDataset(file_list_fake)
    dataset_real = ImageDataset(file_list_real)
    print(dataset_real._path2id)
    # print("??",dataset_real.get_id(r"C:\Users\guodi\Desktop\01_images\images_real\aachen_000026_000019_leftImg8bit.png"))
    data = MatchedCrops(dataset_fake, dataset_real,dataset_fake2, matched_crop_path, weight_path)

    loader = DataLoader(data, batch_size=1, shuffle=True)
    for idx, i in enumerate(loader):
        k = i[0][0]
        j = i[1][0]
        m = i[2][0]
        k = k/255
        k = k.double()
        save_image(k, out_dir / (str(idx)+"label"+".jpg"))
        save_image(j, out_dir / (str(idx) + "real" + ".jpg"))
        save_image(m, out_dir / (str(idx) + "fake" + ".jpg"))


    # input_label = torch.FloatTensor(2, 35, 256, 256).zero_()
    # input_semantics = input_label.scatter_(1, k, 1.0)
    # print(input_semantics.shape)

    # test0 = tf.ToPILImage()(k)
    # test0.show()
    # test1 = tf.ToPILImage()(j)
    # test1.show()
    # test2 = tf.ToPILImage()(m)
    # test2.show()




