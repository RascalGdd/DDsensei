import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16


device = "cpu"
#
# x = torch.randn([1, 3, 128, 64])
# # x = nn.Flatten()(x)
# y = torch.randn([1, 3, 128, 64])
# # y = nn.Flatten()(y)


def MMD(x, y, kernel):
    batch = x.shape[1]
    x = torch.reshape(x, [batch, -1])
    y = torch.reshape(y, [batch, -1])
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)

class Vggextractor():
    def __init__(self) -> None:
        # super().__init__()
        self.vgg = vgg16(pretrained=True, progress=True).features
        self.relu12 = self.vgg[:4]
        self.relu22 = self.vgg[:9]
        self.relu33 = self.vgg[:16]
        self.relu43 = self.vgg[:23]
        self.relu53 = self.vgg[:30]

    def forward(self, x, mode):
        if mode == "relu12":
            return self.relu12(x)
        elif mode == "relu22":
            return self.relu22(x)
        elif mode == "relu33":
            return self.relu33(x)
        elif mode == "relu43":
            return self.relu43(x)
        elif mode == "relu53":
            return self.relu53(x)
        else:
            print("error! please determine from which layer to extract features!")

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class MMD_computer:
    def __init__(self):
        self.extractor = Vggextractor()
        self.mmd_loss = MMD_loss()
        self.flat = nn.Flatten()
    def __call__(self, x, y, mode):
        assert x.shape[0] == 1
        feature_x = self.extractor.forward(x, mode)
        feature_y = self.extractor.forward(y, mode)
        # feature_x, feature_y = self.flat(feature_x), self.flat(feature_y)
        # mmd_loss = self.mmd_loss(feature_x, feature_y)
        mmd_loss = MMD(feature_x, feature_y, "multiscale")
        return mmd_loss * 1000
# print(MMD_computer()(x,y,"relu53"))


# print(MMD_computer()(x,y,"relu53"))



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# from scipy.stats import dirichlet
# from torch.distributions.multivariate_normal import MultivariateNormal
#
#
# m = 20 # sample size
# x_mean = torch.zeros(2)+1
# y_mean = torch.zeros(2)
# x_cov = 2*torch.eye(100) # IMPORTANT: Covariance matrices must be positive definite
# y_cov = 3*torch.eye(100) + 1
#
# px = MultivariateNormal(torch.ones(100), x_cov)
# qy = MultivariateNormal(torch.ones(100), y_cov)
# x = px.sample([1]).to(device)
# y = qy.sample([1]).to(device)
# print(x.shape)
#
# result = MMD(x, x, kernel="multiscale")
#
# print(f"MMD result of X and Y is {result.item()}")
