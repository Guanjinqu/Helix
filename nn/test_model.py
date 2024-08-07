import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, dict):
        super(_netG, self).__init__()
        self.ngpu = dict["ngpu"]
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(dict["nc"],dict["nef"],2,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(dict["nef"],dict["nef"],2,2,1, bias=False),
            nn.BatchNorm2d(dict["nef"]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(dict["nef"],dict["nef"]*2,2,2,1, bias=False),
            nn.BatchNorm2d(dict["nef"]*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(dict["nef"]*2,dict["nef"]*4,2,2,1, bias=False),
            nn.BatchNorm2d(dict["nef"]*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(dict["nef"]*4,dict["nef"]*8,2,2,1, bias=False),
            nn.BatchNorm2d(dict["nef"]*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(dict["nef"]*8,dict["nBottleneck"],2,2 ,bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(dict["nBottleneck"]),
            nn.LeakyReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(dict["nBottleneck"], dict["ngf"] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dict["ngf"] * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(dict["ngf"] * 8, dict["ngf"] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dict["ngf"] * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(dict["ngf"] * 4, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output