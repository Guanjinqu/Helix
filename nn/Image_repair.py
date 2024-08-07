from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from test_model import _netG

import utils

class Image_repair():
    def __init__(self, dict= {
        "nz" : 100,
        "ngf" : 16,
        "ndf" : 16,
        "nc" : 3,
        "ngpu" : 1,
        "netG" : "netG.pth",
        "nBottleneck" : 4000,
        "overlapPred" : 1,
        "nef" : 24
    }):
        self.dict = dict 
        self.netG = _netG(self.dict)
        # netG = TransformerNet()
        self.netG.load_state_dict(torch.load(self.dict["netG"],map_location=lambda storage, location: storage)['state_dict'])
        # netG.requires_grad = False
        self.netG.eval()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.index = 0

    def repair_fuction(self,input_image,i,index):
        """
        Image repair function.

        Parameters:
        - self: instance object.
        - input_image: input image data.
        - i: image sequence number.
        - index: index of the current process.

        Returns:
        - recon_image.data[0]: Reconstructed image data.
        """

        image = self.transform(input_image)
        image = image.repeat(1, 1, 1, 1)

        input_real = torch.FloatTensor(1, 3, 48, 48)
        input_cropped = torch.FloatTensor(1, 3, 48, 48)
        real_center = torch.FloatTensor(1, 3, 16, 16)

        criterionMSE = nn.MSELoss()

        # if opt.cuda:
        #     netG.cuda()
        #     input_real, input_cropped = input_real.cuda(),input_cropped.cuda()
        #     criterionMSE.cuda()
        #     real_center = real_center.cuda()

        input_real = Variable(input_real)
        input_cropped = Variable(input_cropped)
        real_center = Variable(real_center)


        input_real.data.resize_(image.size()).copy_(image)
        input_cropped.data.resize_(image.size()).copy_(image)
        real_center_cpu = image[:,:,16:32,16:32]
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)


        input_cropped.data[:,0,16+int(self.dict["overlapPred"]):int(32-self.dict["overlapPred"]),int(16+self.dict["overlapPred"]):int(32-self.dict["overlapPred"])] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,int(16+self.dict["overlapPred"]):int(32-self.dict["overlapPred"]),int(16+self.dict["overlapPred"]):int(32-self.dict["overlapPred"])] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,int(16+self.dict["overlapPred"]):int(32-self.dict["overlapPred"]),int(16+self.dict["overlapPred"]):int(32-self.dict["overlapPred"])] = 2*123.0/255.0 - 1.0

        fake = self.netG(input_cropped)
        errG = criterionMSE(fake,real_center)

        recon_image = input_cropped.clone()
        recon_image.data[:,:,16:32,16:32] = fake.data

        try:
            os.makedirs("repair/cropped")
            os.makedirs("repair/real")
            os.makedirs("repair/recon")
        except OSError:
            pass

        #utils.save_image("repair/real/"+str(i)+"_"+str(index)+'.png',image[0])
        utils.save_image("repair/cropped/"+str(i)+"_"+str(index)+'.png',image[0]) #input_cropped.data[0]
        utils.save_image("repair/recon/"+str(i)+"_"+str(index)+'.png',recon_image.data[0])
        self.index+=1
        return recon_image.data[0]


