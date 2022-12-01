import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import dataloaders.cropdataset.final_data as final_data
import utils.utils as utils
from utils.fid_scores import fid_pytorch
# from utils.miou_scores import miou_pytorch
from models.models import cfg
import config
from torchmetrics.image.kid import KernelInceptionDistance
import matplotlib.pyplot as plt
import os
import dataloaders.cropdataset_kvd.final_data as final_data_kvd
from utils.mmd import MMD_computer
import numpy
from config import load_iter

#--- read options ---#
opt = config.read_arguments(train=True)
load_iter(opt)

print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
# visualizer_losses = utils.losses_saver(opt)
# losses_computer = losses.losses_computer(opt)
# dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
# if opt.crop:
#     dataloader = final_data.get_dataloader()
# im_saver = utils.image_saver(opt)
# fid_computer = fid_pytorch(opt, dataloader_val)
# miou_computer = miou_pytorch(opt,dataloader_val)

opt.load_size2 = 512
opt.crop_size2 = 512
opt.aspect_ratio2 = 2.0
opt.label_nc = 34
opt.contain_dontcare_label = True
opt.semantic_nc = 35  # label_nc + unknown
opt.cache_filelist_read = False
opt.cache_filelist_write = False
opt.for_metrics = True
opt.load_size = 512
opt.crop_size = 512
opt.label_nc = 34
opt.contain_dontcare_label = True
opt.aspect_ratio = 2.0

kid = KernelInceptionDistance(subset_size=2, reset_real_features=False).cuda()
a, b = [], []

#--- create models ---#
model = models.Unpaired_model(opt, cfg)
model = models.put_on_multi_gpus(model, opt)

#--- create optimizers ---#
# optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
# optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=0.0001,betas=(0.9,0.999), weight_decay=0.0001)
# optimizerD_ori = torch.optim.Adam(model.module.netD_ori.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
# optimizerDu = torch.optim.Adam(model.module.netDu.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))
# optimizerDe = torch.optim.Adam(model.module.wavelet_decoder.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))
def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item

if opt.kvd:
    print("kvd mode!")
    mmd = MMD_computer()


    num_samples = 1800
    total_mmd_loss = 0
    total_mmd_loss2 = 0
    dataloader_kvd = final_data_kvd.get_dataloader_kvd()
    for i, data_i in enumerate(dataloader_kvd):
        real_img, fake_lab = models.preprocess_input_kvd(opt, data_i)
        # print("real_img,",real_img.shape)
        # print("fake_lab,", fake_lab.shape)
        generated = model.module.netEMA(fake_lab)
        # print("generated shape", generated.shape)
        # print(torch.max(generated))
        # print(torch.min(generated))
        generated = generated.detach().cpu()
        real_img = real_img.detach().cpu()
        total_mmd_loss += (mmd(generated, real_img, "relu53").detach().cpu().numpy() - total_mmd_loss) / (i+1)
        total_mmd_loss2 += (mmd(generated, real_img, "relu12").detach().cpu().numpy() - total_mmd_loss2) / (i + 1)
        if i == num_samples:
            break
    # total_mmd_loss = total_mmd_loss / (num_samples+1)
    # total_mmd_loss2 = total_mmd_loss2 / (num_samples + 1)
    print("The KVD is {}".format(total_mmd_loss))
    print("The KVD_12 is {}".format(total_mmd_loss2))