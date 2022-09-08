from pathlib import Path
import os

path_folder_fake = Path("C://Users//guodi//Desktop//01_images//images_fake")
path_folder_fake_label = Path("C://Users//guodi//Desktop//01_images//images_fake_label")
# path_folder_real_1 = Path("/data/public/cityscapes/leftImg8bit")
path_folder_real = Path("C://Users//guodi//Desktop//01_images//images_real")
# city_list = list(path_folder_real.iterdir())

file_list_fake = list(path_folder_fake.iterdir())
file_list_fake_label = list(path_folder_fake_label.iterdir())
file_list_real = list(path_folder_real.iterdir())
# file_list_real = []
# for i in city_list:
#     file_list_real += list((path_folder_real/i).iterdir())




out_dir = Path("C://Users//guodi//Desktop//01_images//cropdata")
fake_path = Path(out_dir/"crop_fake.csv")
real_path = Path(out_dir/"crop_real.csv")
fake_feature_path = Path(out_dir/"crop_fake.npz")
real_feature_path = Path(out_dir/"crop_real.npz")


match_path = Path(out_dir/"match.npz")
weight_path = Path(out_dir/"weight.npz")
matched_crop_path = Path(out_dir/"matched_crop.csv")

