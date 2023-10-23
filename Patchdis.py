import numpy as np
import torch
import torch.nn as nn
class Patch_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Patch_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):

        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)   #因为这里有cat必须是两个大小一样的tensor
        #img_input = torch.cat((img_A, img_B), dim=2)
        return self.model(img_input)
    

netD = Patch_Discriminator()

x_degraded = torch.randn(2, 3, 128, 128)
G_x = torch.randn(2, 3, 128, 128)

y = torch.randn(2, 3, 128, 128)

fake_I = netD(x_degraded, G_x)
print(fake_I)
print(fake_I.shape)  
"""
input:[2,3,102,102] [2,3,102,102]
output:[2,1,6,6]

input:[2,3,128,128] [2,3,128,128]
output:[2,1,8,8]

img_AA不管是多少 img_BB不管是多少
经过一个网络后统一变为[2,3,128,128]

"""

# real_I =netD(x_degraded, y)
# print(real_I.shape)