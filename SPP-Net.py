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
    def __init__(self, num_classes, pool_sizes):
        super(SPPNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 添加你的卷积层和池化层
        )
        self.spp = SPPLayer(pool_sizes)
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
pool_sizes = [(4, 4), (2, 2), (1, 1)]  # 不同尺度的池化大小
sppnet_model = SPPNet(num_classes, pool_sizes)

# 示例：随机输入
input_image = torch.randn(1, 3, 584, 224)  # 1张RGB图像，大小为224x224 torch.Size([1, 63])
output = sppnet_model(input_image)

# 输出的形状
print(output.shape)
