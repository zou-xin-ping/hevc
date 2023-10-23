
import numpy as np
import matplotlib.pyplot as plt
''''
该代码是实验随机缩放的图表绘制
'''
# 启用usetex选项
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # 设置字体为衬线字体，可以根据需要更改字体
# 三个数据集
# data1 = [0.010, 0.032, 0.039, 0.062, 0.084, 0.092, 0.096, 0.104 ]
# data2 = [0.220, 0.314, 0.394, 0.440, 0.434, 0.387, 0.345, 0.313]
# data3 = [0.230, 0.333, 0.413, 0.455, 0.441, 0.399, 0.359, 0.330]
# data1 = [0.107, 0.104, 0.124, 0.062, 0.104, 0.113, 0.121, 0.124 ]
# data2 = [0.212, 0.300, 0.415, 0.440, 0.451, 0.451, 0.375, 0.348]
# data3 = [0.216, 0.361, 0.460, 0.455, 0.511, 0.519, 0.432, 0.398]
data1 = [0.001, 0.001, 0.002, 0.040, 0.008, 0.010, 0.014, 0.016 ]
data2 = [0.195, 0.266, 0.331, 0.366, 0.364, 0.321, 0.291, 0.264]
data3 = [0.220, 0.284, 0.367, 0.412, 0.386, 0.340, 0.311, 0.281]

# x轴坐标
x = [0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 1.7, 1.9]

# 绘制并列直方图
width = 0.05  # 设置每个直方的宽度
plt.bar(x, data1, width, label=r' $\mathcal{M}$$^{D|P}$', color='#A0522D')
plt.bar(np.array(x) + width, data2, width, label=r' $\mathcal{M}$$^{ReLoc}$', color='#008B8B')
plt.bar(np.array(x) + 2 * width, data3, width, label=r' $\mathcal{M}$$^{F–ReLoc}$', color='#FF8C00')

# 设置图例
plt.legend()

# 设置图的标题和轴标签
#plt.title('data')
plt.xlabel(r'\textbf{Scaling factor}')
plt.ylabel(r'\textbf{F1}')
# 设置 x 轴刻度
plt.xticks(x)
# 显示图
plt.show()
