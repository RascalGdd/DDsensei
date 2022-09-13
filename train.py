import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import dataloaders.cropdataset.final_data as final_data
import utils.utils as utils
from utils.fid_scores import fid_pytorch
from utils.miou_scores import miou_pytorch
from models.models import cfg
import config
from torchmetrics.image.kid import KernelInceptionDistance
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR

#--- read options ---#
opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
if opt.crop:
    dataloader = final_data.get_dataloader()
im_saver = utils.image_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)
miou_computer = miou_pytorch(opt,dataloader_val)
kid = KernelInceptionDistance(subset_size=2, reset_real_features=False).cuda()
a, b = [], []

if opt.crop:
    opt.crop_size = 256
    opt.load_size = 286
    opt.aspect_ratio = 1.0

#--- create models ---#
model = models.Unpaired_model(opt,cfg)
model = models.put_on_multi_gpus(model, opt)

#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=0.0001,betas=(0.9,0.999), weight_decay=0.0001)
optimizerD_ori = torch.optim.Adam(model.module.netD_ori.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
optimizerDu = torch.optim.Adam(model.module.netDu.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))
# optimizerDe = torch.optim.Adam(model.module.wavelet_decoder.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))

#Scheduler
scheduler = StepLR(optimizerD, step_size=100000, gamma=0.5)

def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item


#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
if opt.model_supervision != 0 :
    supervised_iter = loopy_iter(dataloader_supervised)
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        scheduler.step()
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        if not opt.crop:
            image, image2, label = models.preprocess_input2(opt, data_i)
        else:
            image, image2, label = models.preprocess_input3(opt, data_i)

        # for m in dataloader_val:
        #     print("1", m["label"].shape)
        #     print("2", m["image"].shape)
        #     break



        # if cur_iter <= 80000:
        # model.module.netG.zero_grad()
        # loss_G, losses_G_list = model(image, label, "losses_G_usis", losses_computer)
        # loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        # loss_G.backward()
        # optimizerG.step()
        #
        # # --- generator conditional update ---#
        # if opt.model_supervision != 0:
        #     supervised_data = next(supervised_iter)
        #     p_image, p_label = models.preprocess_input(opt, supervised_data)
        #     model.module.netG.zero_grad()
        #     p_loss_G, p_losses_G_list = model(image, label, "losses_G_supervised", losses_computer)
        #     p_loss_G, p_losses_G_list = p_loss_G.mean(), [loss.mean() if loss is not None else None for loss in
        #                                                   p_losses_G_list]
        #     p_loss_G.backward()
        #     optimizerG.step()
        # else:
        #     p_loss_G, p_losses_G_list = torch.zeros((1)), [torch.zeros((1))]
        #
        # # --- discriminator update ---#
        # model.module.netD_ori.zero_grad()
        # loss_D, losses_D_list = model(image, label, "losses_D_usis", losses_computer)
        # loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        # loss_D.backward()
        # optimizerD_ori.step()
        #
        # # --- unconditional discriminator update ---#
        model.module.netDu.zero_grad()
        # model.module.wavelet_decoder.zero_grad()
        loss_Du, losses_Du_list = model(image, label, "losses_Du_usis", losses_computer)
        loss_Du, losses_Du_list = opt.reg_every * loss_Du.mean(), [loss.mean() if loss is not None else None for
                                                                   loss in losses_Du_list]
        loss_Du.backward()
        # optimizerDe.step()
        optimizerDu.step()


        # if True:
        #     utils.save_networks(opt, cur_iter, model, latest=True)
        # if True:
        #     is_best = fid_computer.update(model, cur_iter)
        #     if is_best:
        #         utils.save_networks(opt, cur_iter, model, best=True)
        #     _ = miou_computer.update(model,cur_iter)
        #     print("no problem now")
        #     asd





        #
        # # --- generator psuedo labels updates ---@
        #
        # # --- unconditional discriminator regulaize ---#
        if i % opt.reg_every == 0:
            model.module.netDu.zero_grad()
            loss_reg, losses_reg_list = model(image, label, "Du_regulaize", losses_computer)
            loss_reg, losses_reg_list = loss_reg.mean(), [loss.mean() if loss is not None else None for loss in
                                                          losses_reg_list]
            loss_reg.backward()
            optimizerDu.step()
        else:
            loss_reg, losses_reg_list = torch.zeros((1)), [torch.zeros((1))]



        # else:


        #--- generator unconditional update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer,image2)
        loss_G, losses_G_list = loss_G, [loss for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()


        model.module.netG.zero_grad()
        loss_G_ori = model(image, label, "losses_G_ori2", losses_computer,image2)
        loss_G_ori = loss_G_ori.mean()
        loss_G_ori.backward()
        optimizerG.step()


        # --- generator conditional update ---#
        if opt.model_supervision != 0 :
            supervised_data = next(supervised_iter)
            p_image, p_label = models.preprocess_input(opt,supervised_data)
            model.module.netG.zero_grad()
            p_loss_G, p_losses_G_list = model(image, label, "losses_G_supervised", losses_computer, image2)
            p_loss_G, p_losses_G_list = p_loss_G, [None for loss in p_losses_G_list]
            p_loss_G.backward()
            optimizerG.step()
        else :
            p_loss_G, p_losses_G_list = torch.zeros((1)), [torch.zeros((1))]


        #--- discriminator update ---#
        model.module.netD.zero_grad()
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer, image2)
        loss_D, losses_D_list = loss_D, [loss for loss in losses_D_list]
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(model.module.netD.parameters(), 1000.)
        optimizerD.step()

        # model.module.netD_ori.zero_grad()
        # loss_D_ori, losses_D_list_ori = model(image, label, "losses_D_ori", losses_computer)
        # loss_D_ori, losses_D_list_ori = loss_D_ori.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list_ori]
        # loss_D_ori.backward()
        # optimizerD_ori.step()

        #--- unconditional discriminator update ---#
        # model.module.netDu.zero_grad()
        # model.module.wavelet_decoder.zero_grad()
        # model.module.wavelet_decoder2.zero_grad()
        # loss_Du, losses_Du_list = model(image, label, "losses_Du", losses_computer)
        # loss_Du, losses_Du_list = opt.reg_every*loss_Du.mean(), [loss.mean() if loss is not None else None for loss in losses_Du_list]
        # loss_Du.backward()
        # optimizerDe.step()
        # optimizerDe2.step()

        #--- lpips ---@
        # if opt.lpips:
        #     print("lpips mode!")
        #     model.module.netG.zero_grad()
        #     lpips_loss = model(image2, label, "LPIPS", losses_computer)
        #     lpips_loss.backward()
        #     optimizerG.step()
        model.module.netD.zero_grad()
        loss_D_reg, _ = model(image, label, "losses_D_reg", losses_computer, image2)
        loss_D_reg.backward()
        optimizerD.step()


        # --- unconditional discriminator regulaize ---#
        loss_reg, losses_reg_list = torch.zeros((1)), [torch.zeros((1))]
        losses_Du_list = [torch.zeros(1), torch.zeros(1)]

        #--- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        #if cur_iter % opt.freq_save_ckpt == 0:
        #    utils.save_networks(opt, cur_iter, model)
        if cur_iter % 2500 == 0:
            kid.reset()
            model.module.netG.eval()
            if not opt.no_EMA:
                model.module.netEMA.eval()
            with torch.no_grad():
                for i, data_i in enumerate(dataloader_val):
                    image, label = models.preprocess_input(opt, data_i)
                    edges = model.module.compute_edges(image)
                    if opt.no_EMA:
                        generated = model.module.netG(label, edges=edges)
                    else:
                        generated = model.module.netEMA(label, edges=edges)
                    generated = ((generated + 1)*50).type(torch.uint8)
                    image = ((image + 1)*50).type(torch.uint8)
                    # generated = torch.nn.functional.interpolate(generated,scale_factor= 0.5, mode='nearest')
                    kid.update(image, real=True)
                    kid.update(generated, real=False)
                kid_mean, kid_std = kid.compute()
                a.append(cur_iter)
                b.append(kid_mean.cpu())
                fig = plt.figure()
                plt.plot(a, b)
                fig.savefig(os.path.join(opt.checkpoints_dir, opt.name, "KID.png"))
            model.module.netG.train()
            if not opt.no_EMA:
                model.module.netEMA.train()


        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
            _ = miou_computer.update(model,cur_iter)

        # visualizer_losses(cur_iter,losses_G_list+p_losses_G_list+losses_D_list_ori+losses_D_list+losses_reg_list)

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")