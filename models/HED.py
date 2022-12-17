import torch
import torch.nn as nn


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.moduleVggOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 2:3, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 0:1, :, :] * 255.0) - 122.67891434

        tensorInput = torch.cat([tensorBlue, tensorGreen, tensorRed], 1)

        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)

        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

        tensorScoreOne = nn.functional.interpolate(input=tensorScoreOne,
                                                   size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear',
                                                   align_corners=False)
        tensorScoreTwo = nn.functional.interpolate(input=tensorScoreTwo,
                                                   size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear',
                                                   align_corners=False)
        tensorScoreThr = nn.functional.interpolate(input=tensorScoreThr,
                                                   size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear',
                                                   align_corners=False)
        tensorScoreFou = nn.functional.interpolate(input=tensorScoreFou,
                                                   size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear',
                                                   align_corners=False)
        tensorScoreFiv = nn.functional.interpolate(input=tensorScoreFiv,
                                                   size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear',
                                                   align_corners=False)


def define_HED(init_weights_="/no_backups/s1422/DDsensei/pretrained_models/network-bsds500.pytorch"):
    net = HED()
    net = net.cuda()
    for para in net.parameters():
        para.requires_grad = False

    if not init_weights_ == None:
        device = "cuda"
        print('Loading HED model from: %s' % init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(device))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print('load the weights successfully')

    return net
