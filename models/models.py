from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator2 as discriminators
import os
import copy
import torch
from torch.nn import init
import models.losses as losses
from models.CannyFilter import CannyFilter
import yaml
from torch import nn, autograd, optim
import models.vgg16 as vg
from models.discriminator_losses import LSLoss
from models.perceptual_losses import LPIPSLoss as lp


vgg = vg.VGG16().cuda()
config_path ="/no_backups/s1422/DDsensei/train_pfd2cs_ie2.yaml"
with open(config_path) as file:
    cfg = yaml.safe_load(file)
def tee_loss(x, y):
    return x+y, y.detach()

disc_cfg = dict(cfg.get('discriminator', {}))
mdl = discriminators.PerceptualDiscEnsemble(cfg)
run_discs = [True] * len(mdl)

def real_penalty(loss, real_img):
    ''' Compute penalty on real images. '''
    b = real_img.shape[0]
    grad_out = torch.autograd.grad(outputs=loss, inputs=[real_img], create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    reg_loss = torch.cat([g.pow(2).reshape(b, -1).sum(dim=1, keepdim=True) for g in grad_out if g is not None], 1).mean()
    return reg_loss



def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


class Unpaired_model(nn.Module):
    def __init__(self, opt, cfg):
        super(Unpaired_model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netD = discriminators.PerceptualDiscEnsemble(cfg)

            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.gan_loss = LSLoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer, image2):
        # Branching is applied to be compatible with DataParallel
        inv_idx = torch.arange(256 - 1, -1, -1).long().cuda()
        label_gc = torch.index_select(label.clone(), 2, inv_idx)
        image_gc = torch.index_select(image.clone(), 2, inv_idx)
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G":
            vgg_weight = 1
            vgg_loss = lp(net='vgg').cuda()

            loss_G_gan = 0
            loss_G_lpips = 0
            fake = self.netG(label,edges = edges)
            realism_maps = self.netD.forward(img=fake, vgg=vgg, fix_input=True,
                                             run_discs=run_discs)
            for i, rm in enumerate(realism_maps):
                loss_G_gan, _ = tee_loss(loss_G_gan, self.gan_loss.forward_gen(rm[0,:,:,:].unsqueeze(0)).mean())
            del rm
            del realism_maps

            loss_G_lpips, _ = tee_loss(loss_G_lpips,
                                             vgg_weight * vgg_loss.forward_fake(fake, image2)[0])
            loss_G = loss_G_gan + loss_G_lpips

            return loss_G, [0, loss_G_lpips, loss_G_gan, 0]

        if mode == "losses_G_supervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            fake_features = self.netDu(fake,for_features = True)
            real_features = self.netDu(image,for_features = True)

            loss_G_feat = 0
            for real_feat,fake_feat in zip(real_features,fake_features):
                loss_G_feat += self.featmatch(real_feat,fake_feat)

            loss_G += loss_G_feat

            return loss_G,[loss_G_feat]

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

        if mode == "losses_D":
            loss_D = 0
            loss_D_fake = 0
            loss_D_real = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            realism_maps = self.netD.forward(img=fake, vgg=vgg,fix_input=True, run_discs=run_discs)
            for i, rm in enumerate(realism_maps):
                loss_D_fake, _ = tee_loss(loss_D_fake, self.gan_loss.forward_fake(rm).mean())
            del rm
            del realism_maps

            realism_maps = self.netD.forward(img=image, vgg=vgg,
                                       fix_input=False, run_discs=run_discs)
            for i, rm in enumerate(realism_maps):
                loss_D_real += self.gan_loss.forward_real(rm).mean()
            del rm
            del realism_maps
            loss_D = loss_D_real + loss_D_fake

            return loss_D, [loss_D_fake, loss_D_real, 0]


        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake


        if mode == "losses_D_reg":
            reg_weight = 0.03
            loss_D_reg = 0
            image.requires_grad = True
            realism_maps = self.netD.forward(img=image, vgg=vgg, robust_img=image,
                                       fix_input=False, run_discs=run_discs)
            for i, rm in enumerate(realism_maps):
                loss_D_reg += self.gan_loss.forward_real(rm).mean()
            del rm
            del realism_maps

            # loss_D2.backward(retain_graph=True)
            reg_loss, _ = tee_loss(0, real_penalty(loss_D_reg, image))
            # (reg_weight * reg_loss).backward()
            return (reg_weight * reg_loss), [(reg_weight * reg_loss)]


    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

class Unpaired_model_gc(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model_gc, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            if opt.netDu == 'wavelet':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.netDu_gc = discriminators.WaveletDiscriminator(opt)
            else :
                self.netDu = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
                self.netDu_gc = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        inv_idx = torch.arange(256 - 1, -1, -1).long().cuda()
        label_gc = torch.index_select(label.clone(), 2, inv_idx)
        image_gc = torch.index_select(image.clone(), 2, inv_idx)
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G_gc":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            pred_fake = self.netDu(fake)
            fake_gc = self.netG(label_gc, edges=edges)
            pred_fake_gc = self.netDu_gc(fake_gc)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean() + self.criterionGAN(pred_fake_gc, True).mean()

            loss_G_adv = self.get_gc_vf_loss(fake, fake_gc)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            loss_G += loss_G_GAN

            loss_G_vgg = None
            loss_G_edge = None
            return loss_G, [loss_G_adv, loss_G_vgg, loss_G_GAN, loss_G_edge]

        if mode == "losses_Du_gc":
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
                fake_gc = self.netG(label_gc,edges = edges)
            output_Du_fake = self.netDu(fake)
            output_Du_fake_gc = self.netDu_gc(fake_gc)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du_fake_gc = self.criterionGAN(output_Du_fake_gc, False).mean()

            output_Du_real = self.netDu(image)
            output_Du_real_gc = self.netDu_gc(image_gc)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du_real_gc = self.criterionGAN(output_Du_real_gc, True).mean()

            loss_Du = 0.5*(loss_Du_real_gc + loss_Du_fake_gc) + 0.5*(loss_Du_real + loss_Du_fake)

            return loss_Du, [loss_Du_fake,loss_Du_real]


        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake


        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regulaize":
            loss_Du = 0
            image.requires_grad = True
            image_gc.requires_grad = True
            real_pred = self.netDu(image)
            real_pred_gc = self.netDu_gc(image_gc)
            r1_loss = d_r1_loss(real_pred, image).mean()
            r1_loss_gc = d_r1_loss(real_pred_gc, image_gc).mean()
            loss_Du += 10 * (r1_loss + r1_loss_gc)
            return loss_Du, [r1_loss]

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0
        criterionGc = torch.nn.L1Loss()
        size = 256

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * 1.0 #20
        #loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netDu.load_state_dict(torch.load(path + "Du.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netDu_gc, self.netDu]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

class Unpaired_model_cycle(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model_cycle, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator_cycle(opt)
            if opt.netDu == 'wavelet':
                self.netDu_image = discriminators.WaveletDiscriminator(opt)
                self.netDu_label = discriminators.TileStyleGAN2Discriminator(opt.semantic_nc, opt=opt)
            else :
                self.netDu_image = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
                self.netDu_label = discriminators.TileStyleGAN2Discriminator(opt.semantic_nc, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        edges = None

        if mode == "losses_G_cycle":
            loss_G = 0
            # Cycle label->image->label
            fake = self.netG(label,edges = edges)
            pred_fake = self.netDu_image(fake)
            cycle_label = self.netD(fake)

            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

            loss_G_cycle = self.opt.lambda_segment*losses_computer.loss(cycle_label, label, for_real=True)
            #loss_G_cycle = 10.0 * nn.L1Loss(reduction="mean")(cycle_label, label)

            loss_G += loss_G_cycle
            loss_G += loss_G_GAN
            # Cycle 2 image->label->image
            fake_label = self.netD(image)
            pred_fake_label = self.netDu_label(fake_label)
            cycle_image = self.netG(fake_label, None)

            loss_G_GAN2 = self.criterionGAN(pred_fake_label, True).mean()
            loss_G_cycle2 = 10.0 * nn.L1Loss(reduction="mean")(cycle_image, image)
            loss_G += loss_G_GAN2
            loss_G += loss_G_cycle2

            loss_G_vgg = None
            loss_G_edge = None
            return loss_G, [loss_G_cycle, loss_G_vgg, loss_G_GAN, loss_G_edge]

        if mode == "losses_Du_cycle":
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
                fake_label = self.netD(image)

            output_Du_fake = self.netDu_image(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            output_Du_real = self.netDu_image(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()

            output_Du_fake_label = self.netDu_label(fake_label)
            loss_Du_fake_label = self.criterionGAN(output_Du_fake_label, False).mean()
            output_Du_real_label = self.netDu_label(label)
            loss_Du_real_label = self.criterionGAN(output_Du_real_label, True).mean()

            loss_Du = 0.5*(loss_Du_real + loss_Du_fake) + 0.5*(loss_Du_real_label + loss_Du_fake_label)

            return loss_Du, [loss_Du_fake,loss_Du_real]


        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake


        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regulaize":
            loss_Du = 0
            image.requires_grad = True
            label.requires_grad = True
            real_pred = self.netDu_image(image)
            real_pred_label = self.netDu_label(label)
            r1_loss = d_r1_loss(real_pred, image).mean()
            r1_loss_label = d_r1_loss(real_pred_label, label).mean()
            loss_Du += 10 * (r1_loss + r1_loss_label)
            return loss_Du, [r1_loss]

    def gumbelSampler(self, fake, hard=True, eps=1e-10, dim=1):
        #print(fake)
        logits = torch.log(fake + 0.00001)
        if torch.isnan(logits.max()).data:
            print(fake.min(), fake.max())
        if eps != 1e-10:
            print("`eps` parameter is deprecated and has no effect.")

        gumbels = -(torch.empty_like(logits).exponential_() + eps).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / 1.0  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = (y_hard - y_soft).detach() + y_soft
            return index.type(torch.cuda.FloatTensor), ret
        else:
            # Reparametrization trick.
            ret = y_soft
            return 0, ret

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netDu_image.load_state_dict(torch.load(path + "Du_image.pth"))
            self.netDu_label.load_state_dict(torch.load(path + "Du_label.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD, self.netDu_image, self.netDu_label]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to("cuda")
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map

def preprocess_input2(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
        data['image2'] = data['image2'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'],data['image2'], input_semantics
