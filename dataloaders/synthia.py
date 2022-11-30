import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TR
import os
import numpy as np
from PIL import Image
import cv2

synthia_path = os.path.join(r"/data/public/synthia/RAND_CITYSCAPES")
# synthia_path = os.path.join(r"/no_backups/s1422/synthia_part")
id_to_trainid = {
    0: 0,
    1: 23,
    2: 11,
    3: 7,
    4: 8,
    5: 13,
    6: 21,
    7: 17,
    8: 26,
    9: 20,
    10: 24,
    11: 33,
    12: 32,
    13: 9,
    14: 20,
    15: 19,
    16: 22,
    17: 25,
    18: 27,
    19: 28,
    20: 31,
    21: 12,
    22: 7}


class Synthia_data(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.label_path = os.path.join(synthia_path, "GT", "LABELS")
        self.rgb_path = os.path.join(synthia_path, "RGB")
        self.label_list = os.listdir(self.label_path)
        self.path_img = os.path.join("/data/public/cityscapes")
        self.images = []
        for mode in os.listdir(os.path.join(self.path_img, "leftImg8bit")):
            path_img = os.path.join(self.path_img, "leftImg8bit", mode)
            for city_folder in sorted(os.listdir(path_img)):
                cur_folder = os.path.join(path_img, city_folder)
                for item in sorted(os.listdir(cur_folder)):
                    self.images.append(os.path.join(cur_folder, item))

    def __getitem__(self, index):
        name = self.label_list[index % len(self.label_list)]
        path_img = self.images[index % len(self.images)]
        path_label = os.path.join(self.label_path, name)
        path_img2 = os.path.join(self.rgb_path, name)
        img = Image.open(path_img).convert("RGB")
        img2 = Image.open(path_img2).convert("RGB")

        lbl = cv2.imread(path_label, -1)
        lbl = lbl[:, :, 2]
        for i in range(lbl.shape[0]):
            for j in range(lbl.shape[1]):
                if lbl[i, j] in id_to_trainid.keys():
                    lbl[i, j] = id_to_trainid.get(lbl[i, j])
                else:
                    print("no this id")
        lbl = np.array(lbl, dtype=np.float64)
        lbl = torch.tensor(lbl)
        lbl = lbl.unsqueeze(0)
        lbl = TR.Resize([256, 512], interpolation=TR.InterpolationMode.NEAREST)(lbl)
        img = self.transforms(img)
        img2 = self.transforms(img2)

        return {"image": img, "label": lbl, "name": name, "image2": img2}

    def transforms(self, image):
        # resize
        new_width, new_height = (256, 512)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)

        # to tensor
        image = TR.functional.to_tensor(image)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image

    def __len__(self):
        return min(len(self.label_list), len(self.images))

dataset = Synthia_data(synthia_path)
synthia_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# print("len of synthia dataset", len(dataset))

