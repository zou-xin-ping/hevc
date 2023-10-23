import torch
import torch.nn as nn

class SPPLayer(nn.Module):
    def __init__(self, pool_sizes):
        super(SPPLayer, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        num_features = x.size(1)
        x_pooled = []

        for pool_size in self.pool_sizes:
            pooling = nn.AdaptiveMaxPool2d(output_size=pool_size)
            x_pooled.append(pooling(x).view(-1, num_features * pool_size[0] * pool_size[1]))

        x_pooled = torch.cat(x_pooled, 1)
        return x_pooled

class SPPNet(nn.Module):
    def __init__(self, num_classes):
        super(SPPNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 卷积层和池化层来适应任意大小的输入
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 添加其他卷积层和池化层
        )
        self.spp = SPPLayer([(7, 7), (4, 4), (2, 2)])
        # self.fc = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, num_classes)
        # )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.spp(x)
        #x = self.fc(x)
        return x

# 创建SPPNet模型
num_classes = 10  # 根据你的任务设定类别数量
sppnet_model = SPPNet(num_classes)

# 示例：任意大小的输入
input_image = torch.randn(1, 3, 280, 256)  # 1张RGB图像，大小为256x256  #torch.Size([1, 4416])
output = sppnet_model(input_image)

# 输出的形状
print(output.shape)
