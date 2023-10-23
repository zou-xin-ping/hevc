# import torch 
# import torch.nn as nn
# from torchvision.models import resnet34

# my_resnet = resnet34(pretrained=True)

# is_cuda = torch.cuda.is_available()

# if is_cuda:
#     my_resnet = my_resnet.cuda()

# my_resnet = nn.Sequential(*list(my_resnet.children())[:-1])

# #冻结模型的参数，保持预训练的权重不变
# for p in my_resnet.parameters():
#     p.requires_grad =False

# data = 

import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练的ResNet-34模型
resnet34 = models.resnet34(pretrained=True)

# 去掉最后一个线性层
feature_extractor = nn.Sequential(*list(resnet34.children())[:-1])
#冻结模型的参数，保持预训练的权重不变
for p in feature_extractor.parameters():
    p.requires_grad =False
# 设置模型为评估模式（不会进行梯度计算）
feature_extractor.eval()

# 创建自适应平均池化层
# pool = nn.AdaptiveAvgPool2d((3, 224, 224))
# # 生成一个随机的图片张量，大小为(3, 100, 200)
# img = torch.randn(1,3, 100, 200)
# out = pool(img)
# 示例：将图像传递给特征提取器
input_image = torch.randn(1, 3, 333, 224)  # 1张RGB图像，大小为224x224
features = feature_extractor(input_image)

# features现在包含提取的特征
print(features.shape)  # 输出特征的形状

# import os
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# # 定义数据转换，你可以根据需要进行更改
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图像大小
#     transforms.ToTensor(),  # 将图像转换为张量
# ])

# # 定义数据集的根目录
# data_dir = 'path_to_your_dataset_directory'

# # 创建一个数据集
# dataset = ImageFolder(root=data_dir, transform=transform)

# # 创建一个数据加载器
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 示例：遍历数据加载器并处理图像
# for images, labels in dataloader:
#     # 这里可以对每个批次的图像进行处理
#     print(images.shape)  # 输出批次的图像形状
#     print(labels)  # 输出批次的标签

