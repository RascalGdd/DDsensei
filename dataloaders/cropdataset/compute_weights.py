from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filter import load_matching_crops
from cfg import *


width = 957
height = 526

src_crops,_ = load_matching_crops(matched_crop_path)

d = np.zeros((height, width), dtype=np.int32)
print('Computing density...')
for s in tqdm(src_crops): 
	d[s[1]:s[2],s[3]:s[4]] += 1 

print('Computing individual weights...')
w = np.zeros((len(src_crops), 1)) 
for i, s in enumerate(tqdm(src_crops)):
	w[i,0] = np.mean(d[s[1]:s[2],s[3]:s[4]])
	pass

N = np.max(d)
p = N / w
np.savez_compressed(weight_path, w=p)
