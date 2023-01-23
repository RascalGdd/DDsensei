import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from tqdm import tqdm as tqdm
from utils.drn_segment import drn_105_d_miou
from torchmetrics.image.kid import KernelInceptionDistance

kid = KernelInceptionDistance(subset_size=2, reset_real_features=False).cuda()
#--- read options ---#
# opt = config.read_arguments(train=False)

#--- create dataloader ---#
# _,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
# image_saver = utils.results_saver(opt)

#--- create models ---#
# model = models.Unpaired_model(opt)
# model = models.put_on_multi_gpus(model, opt)
# model.eval()

#--- iterate over validation set ---#
# for i, data_i in tqdm(enumerate(dataloader_val)):
#     _, label = models.preprocess_input(opt, data_i)
#     generated = model(None, label, "generate", None)
#     image_saver(label, generated, data_i["name"])

results_dir = ""
name = ""
ckpt_iter = ""

miou = drn_105_d_miou(results_dir, name, ckpt_iter)
print("MIOU =", miou)

