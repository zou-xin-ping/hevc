
import torch
import torch.nn as nn
import torchvision.models as models

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()

        # 特征提取
        self.feature_extractor = models.resnet50(pretrained=True)
        
        # 金字塔池化
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[1, 2, 3, 6])

        # 上采样
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        # 分类层
        self.final_classifier = nn.Conv2d(4096, num_classes, kernel_size=1)

    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)

        # 金字塔池化
        x = self.pyramid_pooling(x)

        # 上采样
        x = self.upsample(x)

        # 分类层
        x = self.final_classifier(x)

        return x

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()

        self.pool_modules = nn.ModuleList()
        for size in pool_sizes:
            self.pool_modules.append(nn.AdaptiveAvgPool2d(size))

    def forward(self, x):
        pool_outs = [pool(x) for pool in self.pool_modules]
        return torch.cat([x] + pool_outs, dim=1)

# 创建PSPNet模型
num_classes = 21  # 根据你的任务设定类别数量
pspnet_model = PSPNet(num_classes)

# 示例：随机输入
input_image = torch.randn(1, 3, 224, 224)  # 1张RGB图像，大小为224x224
output = pspnet_model(input_image)

# 输出的形状
print(output.shape)


# import torch
# import torch.nn as nn

# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels, pool_sizes):
#         super(PyramidPooling, self).__init__()

#         self.pool_modules = nn.ModuleList()
#         for size in pool_sizes:
#             self.pool_modules.append(nn.AdaptiveAvgPool2d(size))

#     def forward(self, x):
#         pool_outs = [pool(x) for pool in self.pool_modules]
#         return torch.cat([x] + pool_outs, dim=1)

# # 示例用法
# in_channels = 3  # 假设输入特征通道数为2048
# pool_sizes = [1, 2, 3, 6]  # 池化的尺寸列表

# # 创建PyramidPooling模块
# pyramid_pooling = PyramidPooling(in_channels, pool_sizes)

# # 示例输入特征图
# input_features = torch.randn(1, in_channels, 1, 1)  # 示例特征图大小为14x14

# # 使用PyramidPooling进行金字塔池化
# output = pyramid_pooling(input_features)

# # 输出的形状
# print(output.shape)
