from pathlib import Path
import os

path_folder_fake = Path("/data/public/gta/images")
path_folder_fake_label = Path("/data/public/gta/labels")
path_folder_real = Path("/data/public/cityscapes/leftImg8bit/train")
city_list = list(path_folder_real.iterdir())

file_list_fake = list(path_folder_fake.iterdir())
file_list_fake_label = list(path_folder_fake_label.iterdir())
file_list_real = []

for i in city_list:
    file_list_real += list((path_folder_real/i).iterdir())
out_dir = Path("/no_backups/s1422/cropdata")
fake_path = Path(out_dir/"crop_fake.csv")
real_path = Path(out_dir/"crop_real.csv")
fake_feature_path = Path(out_dir/"crop_fake.npz")
real_feature_path = Path(out_dir/"crop_real.npz")


match_path = Path(out_dir/"match.npz")
weight_path = Path(out_dir/"weight.npz")
matched_crop_path = Path(out_dir/"matched_crop.csv")

