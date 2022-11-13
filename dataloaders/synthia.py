import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TR
import os
import numpy as np
from PIL import Image
import cv2

synthia_path = os.path.join(r"C:\Users\guodi\Desktop\synthia")
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
        self.RGB_path = os.path.join(synthia_path, "RGB")
        self.label_path = os.path.join(synthia_path, "GT", "LABELS")
        self.RGB_list = os.listdir(self.RGB_path)

    def __getitem__(self, index):
        name = self.RGB_list[index]
        path_rgb = os.path.join(self.RGB_path, name)
        path_label = os.path.join(self.label_path, name)
        img = Image.open(path_rgb).convert("RGB")

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

        return {"image": img, "label": lbl, "name": name}

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
        return len(self.RGB_list)

dataset = Synthia_data(synthia_path)
synthia_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
