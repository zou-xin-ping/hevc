import numpy as np
import torch
import torch.nn as nn
import math

import torch.nn as nn
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_levels=3):
        super(SpatialPyramidPooling, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        N, C, H, W = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = math.ceil((int(H / level), int(W / level)))
            stride = math.floor((int(H / level), int(W / level)))
            pooling = F.max_pool2d(x, kernel_size=kernel_size, stride=stride).view(N, -1)
            #print(pooling.shape)
            pooling_layers.append(pooling)
        return torch.cat(pooling_layers, dim=1)
# def spp(input_tensor,num_levels):
#     N, C, H, W = input_tensor.size()
#     pooling_layers = []
#     for i in range(num_levels):
#         level = i + 1
#         kernel_size = (int(H / level), int(W / level))
#         stride = (int(H / level), int(W / level))
#         pooling = F.max_pool2d(x, kernel_size=kernel_size, stride=stride).view(N, -1)
#         #print(pooling.shape)
#         pooling_layers.append(pooling)
#     return torch.cat(pooling_layers, dim=1)

def spatial_pyramid_pool(previous_conv= torch.randn(8,256,64,64), num_level=3):
    """"
    previous_conv的channels一样大
    """
    pool_out_all =[]
    for i in range(0,num_level):
        
        if i==0:

            pool_out = F.max_pool2d(previous_conv, kernel_size=(13,13), stride=(13,13)).view(8,256,-1)
            print(pool_out.shape)

        if i==1:
            pool_out = F.max_pool2d(previous_conv, kernel_size=(7,7), stride=(6,6)).view(8,256,-1)
            print(pool_out.shape)

        if i==2:
            pool_out = F.max_pool2d(previous_conv, kernel_size=(5,5), stride=(4,4)).view(8,256,-1)
            print(pool_out.shape)


        pool_out_all.append(pool_out)


    return torch.cat(pool_out_all,dim=2)


class my_net(nn.Module):
    def __init__(self, nc=3, ndf=64) -> None:
        super(my_net, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.pool = SpatialPyramidPooling(num_levels=3)
        self.model = nn.Sequential(
            # nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.2, inplace=True)
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
            nn.Conv2d(self.ndf * 8, 3, 4, 1, 0, bias=False),
        )
        

    def forward(self, x):
        output = self.model(x)        
        return output
    
class my_net_U(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(my_net_U, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.block1 = my_net(nc=in_ch, ndf= out_ch)
        self.block2 = SpatialPyramidPooling(num_levels=3)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        return x



        


# batch =8
# x =torch.randn((batch, 3, 102, 128))   #torch.Size([8, 1024, 3, 3])

# net = my_net_U(in_ch=3, out_ch=128)
# spp = SpatialPyramidPooling(num_levels=3)
# y = spp(torch.randn(8,1024,5,5))
# print(y.shape)
# print(net(x).shape)

""""
x =torch.randn((batch, 3, 128, 128))   #torch.Size([8, 1024, 5, 5])
        torch.Size([8, 63315])
x =torch.randn((batch, 3, 102, 102))   #torch.Size([8, 1024, 3, 3])
        torch.Size([8, 65892])
"""

spp = spatial_pyramid_pool(previous_conv= torch.randn(8,256,25,64), num_level=3)

print(spp.shape)