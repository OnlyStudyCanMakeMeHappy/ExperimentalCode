# 导入faiss库和其他需要的模块
import faiss
import numpy as np
import torch

# 定义参考数据集和查询数据集的大小和维度
nb = 1000  # 参考数据集的大小
nq = 100  # 查询数据集的大小
d = 64  # 数据的维度

# 随机生成参考数据集和查询数据集
np.random.seed(1234)  # 设置随机种子，方便复现结果
xb = np.random.random((nb, d)).astype('float32')  # 参考数据集
xq = np.random.random((nq, d)).astype('float32')  # 查询数据集

res = faiss.StandardGpuResources()   # 创建GPU资源
index = faiss.GpuIndexFlatL2(res , d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(xb)
# k + 1 近邻,最相似的一定是自身与自身
k = 4
D , I = index.search(xq, k + 1)
print(D)
print(I)
