import torch.nn as nn

import torch

batch = 8

class my_cnn(nn.Module):
    def __init__(self,nc=3,ndf=64,out_w=64) -> None:
        super(my_cnn,self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.out_w =out_w


        self.conv1 = nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=(3,3), stride=(1,1), bias=False)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.ndf, out_channels=ndf*2, kernel_size=(3,3), stride=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=self.ndf * 2, out_channels=ndf*4, kernel_size=(3,3), stride=(1,1), bias=False)
        self.pool1 = nn.AdaptiveAvgPool2d(self.out_w)
        self.conv4 = nn.Conv2d(self.ndf * 4, 3, 1, 1, 0, bias=False)
        #self.spp =
    def forward(self,input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pool1(x) #torch.Size([8, 256, 1, 1])
        x = self.conv4(x)

        return x
    
class my_block(nn.Module):
    def __init__(self, nc=3, ndf=64, out_w=64) -> None:
        super(my_block,self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.out_w =out_w

        self.block1 = my_cnn(nc=self.nc, ndf=self.ndf, out_w=self.out_w)

    def forward(self,img_A, img_B):
        input1 = self.block1(img_A)
        input2 = self.block1(img_B)

        return input1,input2
    
x = torch.ones(batch,3,128,128)
y = torch.ones(batch,3,102,102)

Net = my_block(nc=3, ndf=64, out_w=64)
p1, p2 =Net(x, y)

mask = torch.randn(batch,1,128,128)
predict = torch.randn(batch,1,102,102)
Net2 = my_block(nc=1, ndf=64, out_w=64)
m1, m2 = Net2(mask,predict)

print(p1.shape, p2.shape)   # retore: torch.Size([8, 3, 64, 64]) torch.Size([8, 3, 64, 64])
print(m1.shape, m2.shape)   # mask: torch.Size([8, 3, 64, 64]) torch.Size([8, 3, 64, 64])
loss = nn.L1Loss() 
lr = 0.001
#
output=loss(p1,p2)

output.backward()