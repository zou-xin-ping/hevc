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
    def __init__(self, num_classes, fixed_input_size):
        super(SPPNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 修改第一层卷积以适应3x224x224的输入
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 添加其他卷积层和池化层
        )
        # 修改SPP层的pool_sizes以匹配输出大小
        spp_pool_sizes = [(int(fixed_input_size[1] / 8), int(fixed_input_size[2] / 8)),
                         (int(fixed_input_size[1] / 16), int(fixed_input_size[2] / 16)),
                         (int(fixed_input_size[1] / 32), int(fixed_input_size[2] / 32))]
        self.spp = SPPLayer(spp_pool_sizes)
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
fixed_input_size = (3, 224, 224)  # 固定的输入大小
sppnet_model = SPPNet(num_classes, fixed_input_size)

# 示例：3x224x224的输入
input_image = torch.randn(1, 3, 224, 224)  # 1张RGB图像，大小为224x224
output = sppnet_model(input_image)

# 输出的形状
print(output.shape)
