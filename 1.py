# import re
# filename = '512_256_9_0.png'
# match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)_.png", filename)
# if match:
#     global width
#     global height
#     global x_blocks
#     global y_blocks
#     width = match.group(1)
#     height = match.group(2)
#     x_blocks = match.group(3)
#     y_blocks = match.group(4)
#     print(width)


# import tensorflow as tf
# tensor = tf.constant([1, 2, 3])  # 1阶张量，即向量
# rank = tf.rank(tensor)
# print(rank.numpy())  # 输出：1
# if tf.rank(tensor) == 0:
#     # 处理标量
#     pass
# elif tf.rank(tensor) == 1:
#     # 处理向量
#     pass
# elif tf.rank(tensor) == 2:
#     # 处理矩阵
#     pass
# # 继续处理更高阶的情况...

'''pytorch中tensor没有名为rank的属性，您可以使用.dim()获取张量的维度数值，或者使用.shape属性来获取张量的形状'''

import torch
tensor = torch.tensor([1, 2, 3])  # 1阶张量，即向量
num_dims = tensor.dim()
print(num_dims)  # 输出：1

import torch

# 创建一个3阶张量，形状为 (2, 3, 4)
tensor_3d = torch.tensor([[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12]],
                         [[13, 14, 15, 16],
                          [17, 18, 19, 20],
                          [21, 22, 23, 24]]])

print(tensor_3d)
print(tensor_3d.shape)  # 输出：torch.Size([2, 3, 4])
print(tensor_3d.dim())   # 输出：3
