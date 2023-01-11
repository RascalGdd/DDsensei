from pathlib import Path
import os

labels = []
# path_lab = Path(r"/data/public/cityscapes/gtFine")
# path_lab_image = Path(r"/data/public/cityscapes/leftImg8bit")
path_lab = Path(r"/no_backups/s1422/output_labels/validation/id")
path_lab_image = Path(r"/data/public/mapillary/validation/images")
for item in sorted(list(path_lab.iterdir())):
    labels.append(Path(path_lab / item))

file_list_real = labels
print(len(file_list_real))

labels_image = []

# for mode in list(path_lab_image.iterdir()):
#     if mode.stem == "test":
#         continue
#     path_lab_image_2 = Path(path_lab_image / mode)
#     for city_folder in sorted(list(path_lab_image_2.iterdir())):
#         cur_folder = Path(path_lab_image_2 / city_folder)
#         for item in sorted(list(cur_folder.iterdir())):
#             labels_image.append(Path(path_lab_image_2 / city_folder / item))

for item in sorted(list(path_lab_image.iterdir())):
    labels_image.append(Path(path_lab_image / item))


file_list_lab_image = labels_image

path_folder_fake = Path("/data/public/gta/labels")

file_list_fake = list(path_folder_fake.iterdir())

out_dir = Path("/no_backups/s1422/cropdata_mapi_kvd")
fake_path = Path(out_dir/"crop_fake.csv")
real_path = Path(out_dir/"crop_real.csv")
fake_feature_path = Path(out_dir/"crop_fake.npz")
real_feature_path = Path(out_dir/"crop_real.npz")


match_path = Path(out_dir/"match.npz")
weight_path = Path(out_dir/"weight.npz")
matched_crop_path = Path(out_dir/"matched_crop.csv")
matched_crop_path_kvd = Path(out_dir/"matched_crop_kvd.csv")