import logging
import os
from pathlib import Path
import random
import imageio
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
import PIL.Image as Image
import cv2

def mat2tensor(mat):
    t = torch.from_numpy(mat).float()
    if mat.ndim == 2:
        return t.unsqueeze(2).permute(2,0,1)
    elif mat.ndim == 3:
        return t.permute(2,0,1)

def _safe_to(a, device):
	return a.to(device, non_blocking=True) if a is not None else None

def _safe_expand(a):
	return a if a is None or a.dim() == 4 else a.unsqueeze(0)

def _safe_cat(s, dim):
	try:
		return torch.cat(s, dim)
	except TypeError:
		return None

class Batch:
	def to(self, device):
		""" Move all internal tensors to specified device. """
		raise NotImplementedError

class ImageBatch(Batch):
    """ Augment an image tensor with identifying info like path and crop coordinates.

	img  -- RGB image
	path -- Path to image
	coords -- Crop coordinates representing the patch stored in img and taken from the path.

	The coords are used for keeping track of the image position for cropping. If we load an image
	and crop part of it, we want to still be able to compute the correct coordinates for the original
	image. That's why we store the coordinates used for cropping (top y, bottom y, left x, right x).
	"""

    def __init__(self, img, path=None, coords=None):
        self.img      = _safe_expand(img)
        self.path     = path
        self._coords  = (0, img.shape[-2], 0, img.shape[-1]) if coords is None else coords
        pass

    def to(self, device):
        return ImageBatch(_safe_to(self.img, device), path=self.path)

    def _make_new_crop_coords(self, r0, r1, c0, c1):
        return (self._coords[0]+r0, self._coords[0]+r1, self._coords[2]+c0, self._coords[2]+c1)

    def crop(self, r0, r1, c0, c1):
        """ Return cropped patch from image tensor(s). """
        coords = self._make_new_crop_coords(r0, r1, c0, c1)
        return ImageBatch(self.img[:,:,r0:r1,c0:c1], path=self.path, coords=coords)

    @classmethod
    def collate_fn(cls, samples):
        imgs          = _safe_cat([s.img for s in samples], 0)
        paths         = [s.path for s in samples]
        return ImageBatch(imgs, path=paths)
    pass

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, transform=None, for_label=False):
        """

        name -- Name used for debugging, log messages.
        img_paths - an iterable of paths to individual image files. Only JPG and PNG files will be taken.
        transform -- Transform to be applied to images during loading.
        """
        self.for_label = for_label
        img_paths = [Path(p[0] if type(p) is tuple else p) for p in img_paths]
        self.paths = sorted([p for p in img_paths if p.is_file() and p.suffix in ['.jpg', '.png']])

        self._path2id = {p: i for i, p in enumerate(self.paths)}
        self.transform = transform

        print('Found {} images.'.format(len(self.paths)))
        pass

    def _load_img(self, path):
        if self.for_label:
            return np.array(Image.open(path))
        else:
            a = np.clip(cv2.imread(str(path)).astype(np.float32) / 255.0, 0.0, 1.0)[:, :, :3]
            a = cv2.resize(a, [957, 526])
            # a = np.transpose(a, (2, 0, 1))
            # print(a.shape)
            return a




    def get_id(self, path):

        return self._path2id.get(Path(path))

    def __getitem__(self, index):

        idx = index % self.__len__()
        path = self.paths[idx]
        img = self._load_img(path)
        if self.transform is not None:
            img = self.transform(img)
            pass

        img = mat2tensor(img)
        return ImageBatch(img, path)

    def get_by_path(self, path):

        img = self._load_img(path)
        if self.transform is not None:
            img = self.transform(img)
            pass

        img = mat2tensor(img)
        return img, path

    def __len__(self):
        return len(self.paths)