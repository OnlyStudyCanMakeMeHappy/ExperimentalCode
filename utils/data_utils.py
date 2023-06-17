from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np
import torch

# 将数据集划分为K等分, 训练集:测试集 = K - 1 : 1, 进一步按照比例将训练划分为相对信息和绝对信息
# random_spilt方法会丢失Dataset的labels等属性 , 所以继承SubSet自定义数据集划分
# 数据集打乱 np.random.permutation
# 训练数据和测试数据采用不同的transform

class CustomSubset(Subset):
    def __init__(self , dataset : Dataset, indices, transform = None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.labels = [self.dataset.labels[index] for index in indices]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        # 调用了父类SubSet的构造方法,所以拥有self.dataset和self.indices属性
        data, label = self.dataset[self.indices[item]]
        if self.transform is not None:
            data = self.transform(data)
        return data , label

class POP(Dataset):
    def __init__(self, dataset, K=10):
        # dataset.data,dateset.targets -> torch
        # 均分K等分, row by row, P(row , 2)
        size = len(dataset)
        split_sizes = [(size + K - 1) // K] * (size % K) + [size // K] * (K - size % K)
        split_dataset = random_split(dataset, split_sizes)
        self.data = self.match_pair(split_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def match_pair(self , data: list[Dataset]) -> list:
        res = []
        cmp = lambda x, y: 1 if x > y else 0
        # data是一个二维列表, 实现(data[i] , data[j])两两配对
        for i in range(len(data) - 1):
            for j in range(i + 1, len(data)):
                res.extend([(x[0], y[0], cmp(x[1], y[1])) for (x, y) in zip(data[i], data[j]) if x[1] != y[1]])
                res.extend([(x[0], y[0], cmp(x[1], y[1])) for (y, x) in zip(data[i], data[j]) if x[1] != y[1]])
        return res


def spilt_dataset(dataset, r: float, tx=None, ty=None):
    sze = len(dataset)
    shuffled_indices = np.random.permutation(sze)
    datasetX = CustomSubset(dataset, shuffled_indices[0: int(sze * r)], tx)
    datasetY = CustomSubset(dataset, shuffled_indices[int(sze * r):], ty)
    return datasetX, datasetY


def process_dataset(dataset: Dataset, train_ratio, abs_ratio, tx = None , ty = None, N = 10):
    # 划分训练集和测试集, 并施以不同的transform
    train_data, test_data = spilt_dataset(dataset, train_ratio, tx, ty)
    # 进一步将训练集划分为绝对信息和相对信息训练集
    abs_train_dataset, rel_train_dataset = spilt_dataset(train_data, abs_ratio)
    # N等分构建偏序对
    # partial_order_pair = POP(rel_train_dataset, N)
    # return (abs_train_dataset, partial_order_pair, test_data)
    return abs_train_dataset,rel_train_dataset, test_data


def match_partial_pairs(batch):
    # 按照标签分类
    # [image , label]
    classes = []
    cls2ind = {}
    for idx, (data, label) in enumerate(batch):
        if cls2ind.get(label) is None:
            cls2ind[label] = []
            classes.append(label)
        cls2ind[label].append(idx)

    n = len(classes)
    pre, succ, labels = [], [], []

    for i in range(n - 1):
        for j in range(i + 1, n):
            a = 1 if classes[i] > classes[j] else 0
            b = a ^ 1
            indicesx, indicesy = cls2ind[classes[i]], cls2ind[classes[j]]
            for (x, y) in zip(indicesx, indicesy):
                # pairs.extend([torch.stack([dataset[x] , dataset[y]] ,dim = 0), torch.stack([dataset[y] , dataset[x]] ,dim = 0)])
                pre.extend([batch[x][0], batch[y][0]])
                succ.extend([batch[y][0], batch[x][0]])
                labels.extend([a, b])
    pre = torch.stack(pre, dim=0)
    succ = torch.stack(succ, dim=0)
    labels = torch.tensor(labels)
    return pre, succ, labels
