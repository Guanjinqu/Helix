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
import numpy as np
from model import _netlocalD,_netG
import utils

from load_data import MyData
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default="train/" , help='path to dataset') #path
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0) #process size
parser.add_argument('--batchSize', type=int, default=64, help='input batch size') #batch size
parser.add_argument('--imageSize', type=int, default=48, help='the height / width of the input image to network')  #image size

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for') 
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002') 
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda') 
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)") 
parser.add_argument('--netD', default='', help="path to netD (to continue training)")  
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints') 
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder') 
parser.add_argument('--overlapPred',type=int,default=1,help='overlapping edges')  
parser.add_argument('--nef',type=int,default=24,help='of encoder filters in first conv layer')  
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight') 
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight') 

opt = parser.parse_args()
print(opt)



try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



data_dir = "train/" 

dataset = MyData(data_dir)
img,target = dataset[0]

print(type(img),target)

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = _netG(opt)
netG.apply(weights_init)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']



netD = _netlocalD(opt)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0


real_center = torch.FloatTensor(opt.batchSize, 3, 16, 16)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


real_center = Variable(real_center)


optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


writer = SummaryWriter("logs")


writer.add_graph(netG,input_cropped)

for epoch in range(resume_epoch,opt.niter):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        #Selection of the central area
        real_center_cpu = real_cpu[:,:,int(opt.imageSize/3):int(opt.imageSize/3)+16,int(opt.imageSize/3):int(opt.imageSize/3)+16]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

        #Masking the center area
        input_cropped.data[:,0,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*123.0/255.0 - 1.0

  
        netD.zero_grad()
        label.data.resize_(batch_size).fill_(real_label)
         
        output = netD(real_center)
        output = output.squeeze(-1)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()


        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        output = output.squeeze(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        output = output.squeeze(-1)
        errG_D = criterion(output, label)

        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
        wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/3 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/3 - opt.overlapPred)] = wtl2

        errG_l2 = (fake-real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()


        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG_D.item(),errG_l2.item(), D_x,D_G_z1, ))
            vutils.save_image(real_cpu,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            recon_image.data[:,:,int(opt.imageSize/3):int(opt.imageSize/3+16),int(opt.imageSize/3):int(opt.imageSize/3+16)] = fake.data
            vutils.save_image(recon_image.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))


    # do checkpointing
    torch.save({'epoch':epoch+1,
                'state_dict':netG.state_dict()},
                'model/netG_streetview.pth' )
    torch.save({'epoch':epoch+1,
                'state_dict':netD.state_dict()},
                'model/netlocalD.pth' )

writer.close()