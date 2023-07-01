import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
from torchvision.datasets import ImageFolder
#
#
__all__ = ['MPerClassSampler']
# 先随机选取k个类别,然后每个类别随机选取min(len(cls[k] ,p)个样本
class MPerClassSampler(BatchSampler):
    # labels可以是ImageFloder, 也可以是所有数据的标签
    # imgs对象:[(img_path, class_index)]
    @staticmethod
    def labels_to_indices(objection):
        labels2indices = defaultdict(list)
        if isinstance(objection, ImageFolder):
            for idx, (_, label) in enumerate(objection.imgs):
                labels2indices[label].append(idx)
        elif isinstance(objection, torch.Tensor):
            for idx, label in enumerate(objection.numpy().tolist()):
                labels2indices[label].append(idx)
        else:
            for idx, label in enumerate(objection):
                labels2indices[label].append(idx)
        return labels2indices
    def __init__(self, obj, batch_size, m, iter_per_epoch):
        self.m = m
        self.batch_size = batch_size
        self._iter_per_epoch = iter_per_epoch
        self.labels2indices = self.labels_to_indices(obj)
        # if isinstance(obj, (ImageFolder,torch.Tensor)):
        #     self.labels2indices = labels_to_indices(obj)
        # else:
        #     raise ValueError("The objection must be ImageFloder or torch.Tensor")
        self.class_idx = list(self.labels2indices)

    def __len__(self):
        # 返回的是
        return self._iter_per_epoch

    def __iter__(self):
        for _ in range(self._iter_per_epoch):
            # 重排列所有类别, 打乱顺序
            # 固定了随机种子, 每个iter打乱之后的顺序都不相同
            # 随机种子确定了随机生成算法的起点, 每次程序运行生成的序列具有规律性
            np.random.shuffle(self.class_idx)
            selected_indices = []
            # 每个类别最多选取min(m , n), n是类别的数量
            for c in self.class_idx:
                selected = np.random.choice(self.labels2indices[c], size = min(self.m, len(self.labels2indices[c])), replace = False)
                selected_indices.extend(selected)
                if len(selected_indices) >= self.batch_size:
                    break

            yield selected_indices[:self.batch_size]



