
#https://blog.csdn.net/qq_34769162/article/details/115567093
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torchvision.transforms as transforms
from PIL import Image

class SaveOutput:
	def __init__(self):
		self.outputs = []
	def __call__(self, module, module_in, module_out):
		self.outputs.append(module_out)
	def clear(self):
		self.outputs=[]
		
save_output = SaveOutput()

hook_handles = []

for layer in feature_extractor.modules():
	if isinstance(layer, torch.nn.Conv2d):
		handle = layer.register_forward_hook(save_output)
		hook_handles.append(handle)


# 加载预训练的VGG模型
vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) # 新版本的写   (pretrained=True)

# 提取VGG的卷积组件作为特征提取器
feature_extractor = nn.Sequential(*list(vgg_model.features.children()))

# 冻结特征提取器中的参数
for param in feature_extractor.parameters():
    param.requires_grad = False

# 输出特征提取器的结构
print(feature_extractor)
# 读取图片
image_path = r"D:\1第三学期\c992737_0.png"
image = Image.open(image_path)

# 定义转换操作
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # 调整图片大小为224x224
    transforms.ToTensor(),  # 转换为torch Tensor
])

# 应用转换操作
input_image = transform(image)

# 添加批次维度
input_image = input_image.unsqueeze(0)

# 输出输入图片的形状
print(input_image.shape)
# 示例：使用特征提取器提取特征
#input_image = torch.randn(1, 3, 2048, 4500)  # 1张RGB图像，大小为224x224  torch.Size([1, 512, 7, 7])  torch.Size([1, 512, 64, 140])
features = feature_extractor(input_image)

# 输出特征的形状
print(features.shape)


# # 假设x是一个4维张量，形状为(batch_size, channels, height, width)
# x = torch.randn(2, 3, 224, 224)  # 示例：2张RGB图像，大小为224x224

# # 定义转换操作
transform2 = transforms.ToPILImage()

# 保存每张图像
for i in range(features.shape[0]):
    image = transform2(features[i])
    image.save(f"image_{i}.jpg")

# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# # 假设features是一个4维张量，形状为(batch_size, channels, height, width)
# features = torch.randn(2, 64, 112, 112)  # 示例：2张特征图，通道数为64，大小为112x112

# # 取出第一张特征图
# feature_map = features[0]

# # 转换为numpy数组
# feature_map = feature_map.detach().numpy()

# # 将通道维度放在最后
# feature_map = feature_map.transpose((1, 2, 0))

# # 创建一个子图
# fig, ax = plt.subplots()

# # 显示特征图
# ax.imshow(feature_map)

# # 关闭坐标轴
# ax.axis('off')

# # 保存特征图
# plt.savefig("feature_map.jpg", bbox_inches='tight', pad_inches=0)

# # 显示特征图
# plt.show()


