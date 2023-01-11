import csv
import logging
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
import torch

from dataloaders.cropdataset_kvd.batch_types import JointEPEBatch
from dataloaders.cropdataset_kvd.sample_matches import load_crops
from dataloaders.cropdataset_kvd.filter import load_matching_crops
import torchvision.transforms as tf
from torchvision.transforms.functional import InterpolationMode
from dataloaders.cropdataset_kvd.cfg2 import *

def process_path(path: Path):
	# city = str(path).split("/")[-2]
	# name = str(path).split("/")[-1]
	# cut_idx = name.find("gtFine")
	# name2 = name[:cut_idx] + "leftImg8bit.png"
	name = str(path).split("/")[-1].replace(".png", ".jpg")

	# if "train" in str(path):
	# 	path_cityscape_images = Path(path_lab_image / "train" / city / name2)
	# elif "val" in str(path):
	# 	path_cityscape_images = Path(path_lab_image / "val" / city / name2)
	# else:
	# 	print("file path not correct")
	# 	return
	path_cityscape_images = Path(path_lab_image / name)

	return str(path_cityscape_images)


class PairedDataset(torch.utils.data.Dataset):
	def __init__(self, source_dataset, target_dataset, source_dataset2):
		self._source_dataset = source_dataset
		self._target_dataset = target_dataset
		self._source_dataset2 = source_dataset2
		self.src_crops = []
		self.dst_crops = []
		self.src_crops2 = []

		pass


	def _get_cropped_items(self, idx, jdx):
		s = self.src_crops[idx]
		s2 = self.src_crops2[idx]
		t = self.dst_crops[jdx]

		src_id = self._source_dataset.get_id(s[0])
		# print("src_id",src_id)

		dst_id = self._target_dataset.get_id(t[0])
		src_id2 = self._source_dataset2.get_id(s2[0])

		img_fake = torch.squeeze(self._source_dataset[src_id].crop(*s[1:]).img)
		img_real = torch.squeeze(self._target_dataset[dst_id].crop(*t[1:]).img)
		img_real = img_real.unsqueeze(dim=0)
		img_fake2 = torch.squeeze(self._source_dataset2[src_id2].crop(*s2[1:]).img)
		img_fake2 = img_fake2.unsqueeze(dim=0)
		img_fake = tf.Resize([256, 256], interpolation=InterpolationMode.NEAREST)(img_fake)
		img_real = tf.Resize([256, 256], interpolation=InterpolationMode.NEAREST)(img_real).type(torch.int64)
		img_fake2 = tf.Resize([256, 256], interpolation=InterpolationMode.NEAREST)(img_fake2).type(torch.int64)

		# return JointEPEBatch(self._source_dataset[src_id].crop(*s[1:]), self._target_dataset[dst_id].crop(*t[1:]))
		return img_fake, img_real, img_fake2

	def __len__(self):
		return len(self.src_crops)


	@property
	def source(self):
		return self._source_dataset


	@property
	def target(self):
		return self._target_dataset


class MatchedCrops(PairedDataset):
	def __init__(self, source_dataset, target_dataset, source_dataset2, matched_crop_path, crop_weight_path):
		super().__init__(source_dataset, target_dataset, source_dataset2)

		self._weighted = False

		self.src_crops, self.dst_crops = load_matching_crops(matched_crop_path)

		valid_src_crops, valid_dst_crops, valid_src_crops2 = [], [], []
		valid_ids = []
		for i, (sc, dc) in tqdm(enumerate(zip(self.src_crops, self.dst_crops))):
			sc2 = sc
			sc = (process_path(Path(dc[0])), dc[1], dc[2], dc[3], dc[4])

			# print("sc", sc[0])
			# print("dc", dc[0])
			# print("sc2", sc2[0])

			# sc = ((str(path_folder_fake)+"\\"+sc[0].split("\\")[-1]),sc[1],sc[2],sc[3],sc[4])
			# sc = ((str(path_folder_fake_label) + "/" + sc[0].split("/")[-1]), sc[1], sc[2], sc[3], sc[4])
			if self._source_dataset.get_by_path(sc[0]) is not None:
				# print("GO TO not none")
				valid_src_crops.append(sc)
				valid_dst_crops.append(dc)
				valid_src_crops2.append(sc2)
				valid_ids.append(i)
				pass
			pass

		print('Done to {} crops.'.format(len(valid_ids)))

		self.src_crops = valid_src_crops
		self.dst_crops = valid_dst_crops
		self.src_crops2 = valid_src_crops2
		if crop_weight_path is not None:
			d = np.load(crop_weight_path)
			w = d['w']
			w = w[valid_ids]
			self._cumsum = np.cumsum(w) / np.sum(w)
			assert len(self.src_crops) == self._cumsum.shape[0], f'Weights ({self._cumsum.shape[0]}) and source crops ({len(self.src_crops)}) do not match.'
			self._weighted = True
			pass

		print('Sampling Initialized.')
		pass

	def __getitem__(self, idx):
		# try:
		if self._weighted:
			p   = random.random()
			idx = np.min(np.nonzero(p<self._cumsum)[0])
			pass
		return self._get_cropped_items(idx, idx)
		# except KeyError:
		# 	return self.__getitem__(random.randint(0, len(self.src_crops)-1))

	def __len__(self):
		return len(self.src_crops)


class IndependentCrops(PairedDataset):
	def __init__(self, source_dataset, target_dataset, cfg):
		super(IndependentCrops, self).__init__(source_dataset, target_dataset)

		self._crop_size = 196
		pass

	def _sample_crop(self, batch):
		r1 = random.randint(self._crop_size, batch.img.shape[-2])
		r0 = r1 - self._crop_size
		c1 = random.randint(self._crop_size, batch.img.shape[-1])
		c0 = c1 - self._crop_size
		return batch.crop(r0, r1, c0, c1)


	def __getitem__(self, idx):
		return self._sample_crop(self._source_dataset[idx]), \
			self._sample_crop(self._target_dataset[random.randint(0, len(self._target_dataset)-1)])


	def __len__(self):
		return len(self._source_dataset)

		
