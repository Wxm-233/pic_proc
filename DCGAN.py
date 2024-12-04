# check the necessary dependencies
from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# check PyTorch version and cuda status
# print(torch.__version__, torch.cuda.is_available())

# # set random seed for reproducibility
# manualSeed = 999
# #manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# number of channels in the training images. For color images this is 3
nc = 3

# size of z latent vector (i.e. size of generator input)
nz = 100

# size of feature maps in generator
ngf = 64

# generator code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution. nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
# 创建CPU版本的生成器
device = 'cpu'
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load('./dcgan_checkpoint.pth', weights_only=True))
v_a = torch.load('v_a.pt', weights_only=True)

def generate_image(seed, gender):
    random.seed(seed)
    torch.manual_seed(seed)
    test_batch_size = 1
    noise = torch.randn(test_batch_size, nz, 1, 1, device=device)
    with torch.no_grad():
        if gender == '男':
            noise += v_a
        if gender == '女':
            noise -= v_a
        fake = netG(noise).detach().cpu()
    toPIL = transforms.ToPILImage()
    return toPIL(fake[0])
