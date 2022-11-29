import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import config

gtav_dataroot = '/data/public/gta/labels'
gtav_dataroot2 = '/data/public/gta/images'
gtav_dataroot_origin = '/data/public/gta'
opt = config.read_arguments(train=False)

class GTA_VAL(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics, for_supervision = False):

        opt.load_size =  512 if for_metrics else 512
        opt.crop_size =  512 if for_metrics else 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False

        self.opt = opt
        self.for_metrics = for_metrics
        self.for_supervision = False
        self.images, self.labels,self.images2, self.paths = self.list_images()

        if opt.mixed_images and not for_metrics :
            self.mixed_index=np.random.permutation(len(self))
        else :
            self.mixed_index=np.arange(len(self))

        if for_supervision :

            if opt.model_supervision == 0 :
                return
            elif opt.model_supervision == 1 :
                self.supervised_indecies = np.array(np.random.choice(len(self),opt.supervised_num),dtype=int)
            elif opt.model_supervision == 2 :
                self.supervised_indecies = np.arange(len(self),dtype = int)
            images = []
            labels = []

            for index in self.supervised_indecies :
                images.append(self.images[index])
                labels.append(self.labels[index])

            self.images = images
            self.labels = labels

            self.mixed_index = np.arange(len(self))

            classes_counts = np.zeros((34),dtype=int)
            supervised_classes_in_images = []
            counts_in_images = []
            self.weights = []
            for i in tqdm(range(len(self))):
                label = self.__getitem__(i)['label']
                supervised_classes_in_image,counts_in_image = torch.unique(label,return_counts = True)
                supervised_classes_in_image = supervised_classes_in_image.int().numpy()
                counts_in_image = counts_in_image.int().numpy()
                supervised_classes_in_images.append(supervised_classes_in_image)
                counts_in_images.append(counts_in_image)
                for supervised_class_in_image,count_in_image in zip(supervised_classes_in_image,counts_in_image):
                    classes_counts[supervised_class_in_image]+=count_in_image

            for i in range(len(self)):
                weight = 0
                for class_in_image,class_count_in_image in zip(supervised_classes_in_images[i],counts_in_images[0]) :
                    if class_count_in_image != 0 and class_in_image != 0 :
                        weight += class_in_image/classes_counts[class_in_image]

                self.weights.append(weight)

            min_weight = min(self.weights)
            self.weights = [ weight/min_weight for weight in self.weights ]
            self.for_supervision = for_supervision


    def __len__(self,):
        return max(len(self.images),len(self.labels))

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[self.mixed_index[idx]%len(self.images)])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx%len(self.labels)]))
        image2 = Image.open(os.path.join(self.paths[2], self.images2[idx % len(self.labels)]))
        image, label, image2 = self.transforms(image, label, image2)
        label = label * 255
        if self.for_supervision:
            return {"image": image, "label": label, "name": self.images[self.mixed_index[idx]], "weight": self.weights[self.mixed_index[idx]]}
        else:
            return {"image": image, "label": label, "name": self.images[self.mixed_index[idx]%len(self.images)]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        images2 = []
        path_lab = os.path.join(gtav_dataroot)
        for label_map in sorted(os.listdir(path_lab)):
            if label_map.find(".png") != -1:
                labels.append(label_map)
                images2.append(label_map)

        path_img2 = os.path.join(gtav_dataroot2)

        print("different len of images and labels %s - %s" % (len(images), len(labels)))

        return images, labels, images2, (path_img, path_lab, path_img2)

    def transforms(self, image, label, image2):
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        image2 = TR.functional.resize(image2, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # to tensor
        image = TR.functional.to_tensor(image)
        image2 = TR.functional.to_tensor(image2)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image2 = TR.functional.normalize(image2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, image2

gta_val = GTA_VAL(opt, for_metrics=True)