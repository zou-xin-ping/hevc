import torch.nn as nn
import torch
class Discriminator(nn.Module):
    '''
    判别器网络
    '''
    def __init__(self,nc = 3,ndf = 64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ndf * 8, 1, 1, 1, 0, bias=False),

            nn.Sigmoid()      #H:\image_reloc\sppnet-pytorch-master\DCdis.py
        )

    def forward(self, input):
        return self.main(input)
    
dis = Discriminator(nc=3,ndf=64)
batch =2   
data_x = torch.randn((batch,3,102,101)) 
D_real_B =dis(data_x)   #标量值  把这个值和 0 1比较
print(D_real_B)   