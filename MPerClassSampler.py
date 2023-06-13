import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from torchvision.datasets import ImageFolder
#
#
def labels_to_indices(objection):
    labels2indices = defaultdict(list)
    if isinstance(objection, ImageFolder):
        for idx , (_, label) in enumerate(objection.imgs):
            labels2indices[label].append(idx)
    elif isinstance(objection, torch.Tensor):
        for idx, label in enumerate(objection.numpy().tolist()):
            labels2indices[label].append(idx)
    else:
        for idx, label in enumerate(objection):
            labels2indices[label].append(idx)
    return labels2indices

# 先随机选取k个类别,然后每个类别随机选取min(len(cls[k] ,p)个样本
class MPerClassSampler(Sampler):
    # labels可以是ImageFloder, 也可以是所有数据的标签
    # imgs对象:[(img_path, class_index)]
    def __init__(self, obj, batch_size, m, iter_per_epoch):
        self.m = m
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch
        self.labels2indices = labels_to_indices(obj)
        # if isinstance(obj, (ImageFolder,torch.Tensor)):
        #     self.labels2indices = labels_to_indices(obj)
        # else:
        #     raise ValueError("The objection must be ImageFloder or torch.Tensor")
        self.class_idx = list(self.labels2indices)

    def __len__(self):
        # 返回的是
        return self.iter_per_epoch


    def __iter__(self):
        for _ in range(self.iter_per_epoch):
            # 重排列所有类别, 打乱顺序
            np.random.shuffle(self.class_idx)
            selected_indices = []
            # 每个类别最多选取min(m , n), n是类别的数量
            for c in self.class_idx:
                selected = np.random.choice(self.labels2indices[c], size = min(self.m, len(self.labels2indices[c])), replace = False)
                selected_indices.extend(selected)
                if len(selected_indices) >= self.batch_size:
                    selected_indices = selected_indices[:self.batch_size]
                    break

            yield from selected_indices







def index_dataset(dataset: ImageFolder):
    kv = [(cls_ind, idx) for idx, (_, cls_ind) in enumerate(dataset.imgs)]
    cls_to_ind = {}

    for k, v in kv:
        if k in cls_to_ind:
            cls_to_ind[k].append(v)
        else:
            cls_to_ind[k] = [v]

    return cls_to_ind


class MImagesPerClassSampler:
    def __init__(self, data_source: ImageFolder, batch_size, m=5, iter_per_epoch=100):
        self.m = m
        self.batch_size = batch_size
        self.n_batch = iter_per_epoch
        self.class_idx = list(data_source.class_to_idx.values())
        self.images_by_class = index_dataset(data_source)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            # selected_class = random.sample(self.class_idx, k=len(self.class_idx))
            selected_class = np.random.choice(self.class_idx, size=len(self.class_idx), replace=False)
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]

                # maybe not satisfied P*K
                # new_ind = random.sample(img_ind_of_cls, k=min(self.m, len(img_ind_of_cls)))
                new_ind =  np.random.choice(img_ind_of_cls, size=min(self.m, len(img_ind_of_cls)), replace=False).tolist()
                example_indices += new_ind

                if len(example_indices) >= self.batch_size:
                    break

            yield from example_indices[: self.batch_size]