from torch.utils.data.dataset import T_co
import torchvision.transforms as tf
import utils.utils as utils
import config
from tqdm import tqdm as tqdm
from utils.drn_segment import drn_105_d_miou
from torchmetrics.image.kid import KernelInceptionDistance
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import os
import torch
from utils.fid_scores import fid_pytorch_test


kid = KernelInceptionDistance(subset_size=500, reset_real_features=False).cuda()

path_fake = r"/no_backups/s1422/F-LSeSim/results/city_test/1234/image"
path_real = r"/data/public/cityscapes/leftImg8bit/val"

results_dir = r"/no_backups/s1422/F-LSeSim/results"
name = r"city_test"
ckpt_iter = r"1234"


class RealDataset(Dataset):
    def __init__(self, path_img, path_lbl=None):
        self.path_img = path_img
        # self.path_lbl = path_lbl
        self.img_list = []
        for city in os.listdir(self.path_img):
            city_path = os.path.join(self.path_img, city)
            for file in os.listdir(city_path):
                self.img_list.append(os.path.join(city_path, file))

    def __getitem__(self, index):
        img_filepath = self.img_list[index]
        # lbl_filepath = os.path.join(self.path_lbl, self.img_list[index])
        img = Image.open(img_filepath).convert("RGB")
        # lbl = Image.open(lbl_filepath)
        img = tf.Compose([tf.ToTensor(), tf.Resize([256, 512])])(img)
        return img

    def __len__(self):
        return len(self.img_list)

class FakeDataset(Dataset):
    def __init__(self, path_img, path_lbl=None):
        self.path_img = path_img
        # self.path_lbl = path_lbl
        self.img_list = sorted(os.listdir(self.path_img))
        # self.lbl_list = sorted(os.listdir(self.path_lbl))

    def __getitem__(self, index):
        img_filepath = os.path.join(self.path_img, self.img_list[index])
        # lbl_filepath = os.path.join(self.path_lbl, self.img_list[index])
        img = Image.open(img_filepath).convert("RGB")
        # lbl = Image.open(lbl_filepath)
        img = tf.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.img_list)

city_set = RealDataset(path_real)
gta_set = FakeDataset(path_fake)
city_loader = DataLoader(city_set, batch_size=1, shuffle=False)
gta_loader = DataLoader(gta_set, batch_size=1, shuffle=False)

#--- iterate over validation set ---#
# for i, data_i in tqdm(enumerate(dataloader_val)):
#     _, label = models.preprocess_input(opt, data_i)
#     generated = model(None, label, "generate", None)
#     image_saver(label, generated, data_i["name"])

#
miou = drn_105_d_miou(results_dir, name, ckpt_iter)
print("MIOU =", miou)

for i, image in enumerate(city_loader):
    image = (image * 255).type(torch.uint8).cuda()
    kid.update(image, real=True)

for i, generated in enumerate(gta_loader):
    generated = (generated * 255).type(torch.uint8).cuda()
    kid.update(generated, real=False)

kid_mean, kid_std = kid.compute()
print("KID =", kid_mean)

opt = config.read_arguments(train=False)
fid_computer = fid_pytorch_test(opt, city_loader, gta_loader)
fid = fid_computer.update()
print("FID =", fid)
