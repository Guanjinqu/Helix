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
parser.add_argument('--dataroot',  default="train/" , help='path to dataset') #数据类型
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0) #数据加载进程数量
parser.add_argument('--batchSize', type=int, default=64, help='input batch size') #batch的大小
parser.add_argument('--imageSize', type=int, default=48, help='the height / width of the input image to network')  #图片输入的尺寸，这个要改很困难

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for') #迭代次数
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002') #学习率
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5') #adam参数
parser.add_argument('--cuda', action='store_true', help='enables cuda') #是否cuda
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use') #用几个GPU
parser.add_argument('--netG', default='', help="path to netG (to continue training)")  #外置G网络的参数
parser.add_argument('--netD', default='', help="path to netD (to continue training)")  #外置D网络的参数
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints') #放置输出图像和模型参数的位置
parser.add_argument('--manualSeed', type=int, help='manual seed') #随机种子

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder') #编码器bottleneck的维数
parser.add_argument('--overlapPred',type=int,default=1,help='overlapping edges')  #重叠边缘？
parser.add_argument('--nef',type=int,default=24,help='of encoder filters in first conv layer')  #卷积第一次的尺寸
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight') #0表示不要使用这个权重
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight') #0表示不要使用这个权重

opt = parser.parse_args()
print(opt)


#构建 结果文件夹
try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

#设置随机种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

#通过设置cudnn.benchmark 可以使CUDA在卷积之前选择合适的卷积算法，进而加速。
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#导入数据集（待改）

data_dir = "train/" 

dataset = MyData(data_dir)
img,target = dataset[0]

print(type(img),target)
#校验是否导入数据集
assert dataset
#加载数据
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)

#好像以下三个没用上
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10

# 在 netG 和 netD 上调用自定义权重初始化
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
#检测是否需要载入参数
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
#print(netG)


netD = _netlocalD(opt)
netD.apply(weights_init)
#检测是否需要载入参数
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
#print(netD)

#定义损失函数
criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()
#创造一个指定大小的tensor空间
input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

#正确的中心的tensor空间
real_center = torch.FloatTensor(opt.batchSize, 3, 16, 16)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()

#Variable类是autograd包中很重要的一类，它的作用是包装Tensor，将一个tensor其变成计算图中的一个节点(变量)
input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


real_center = Variable(real_center)

# 设置优化器
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


writer = SummaryWriter("logs")

#writer.add_graph(netD,real_center)
writer.add_graph(netG,input_cropped)
#开始正式训练
for epoch in range(resume_epoch,opt.niter):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        #选出中心区域
        real_center_cpu = real_cpu[:,:,int(opt.imageSize/3):int(opt.imageSize/3)+16,int(opt.imageSize/3):int(opt.imageSize/3)+16]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

        #这一步是对中心区域进行遮蔽
        input_cropped.data[:,0,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred),int(opt.imageSize/3+opt.overlapPred):int(opt.imageSize/3+16-opt.overlapPred)] = 2*123.0/255.0 - 1.0



        # 基于真实的训练    
        netD.zero_grad()
        label.data.resize_(batch_size).fill_(real_label)
         
        output = netD(real_center)
        output = output.squeeze(-1)
        #print(output)
        #print(label)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        #生成图像

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
        # errG_D.backward(retain_variables=True)

        # errG_l2 = criterionMSE(fake,real_center)
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
        wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/3 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/3 - opt.overlapPred)] = wtl2
        #print(fake.shape,real_center.shape)
        errG_l2 = (fake-real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()
        #print(errD)
        #print(D_G_z1)

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