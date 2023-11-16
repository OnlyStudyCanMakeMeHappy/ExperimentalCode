import numpy as np
import torch
import random
import  torchvision.transforms as transforms

from collections import defaultdict
from torch.utils.data import  Dataset, random_split, Subset, BatchSampler, DataLoader
from MPerClassSampler import MPerClassSampler
from math import ceil
from typing import *

# 将数据集划分为K等分, 训练集:测试集 = K - 1 : 1, 进一步按照比例将训练划分为相对信息和绝对信息
# random_spilt方法会丢失Dataset的labels等属性 , 所以继承SubSet自定义数据集划分
# 数据集打乱 np.random.permutation
# 训练数据和测试数据采用不同的transform
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self , val , n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    # getitem只接收一个参数
    def __getitem__(self, idx):
        task_id, item = idx
        return self._datasets[task_id][item]

class MTBatchSampler(BatchSampler):

    def __init__(self, datasets, batch_size):
        self.batch_size = batch_size
        self._datasets = datasets
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list
        self._task_seq = self._gen_task_indices(self._train_data_list)

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def getTaskSeq(self):
        return self._task_seq
    def __iter__(self):
        # 获取每个task数据集的迭代器
        all_iters = [iter(item) for item in self._train_data_list]
        # 通过all_indices获取对应index数据集的batch
        for local_task_idx in self._task_seq:
            # 假设local_task_idx就是task_id
            batch = next(all_iters[local_task_idx])
            yield [(local_task_idx, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list):
        main_indices = [0] * len(train_data_list[0])
        extra_indices = []
        for i in range(1, len(train_data_list)):
            extra_indices += [i] * len(train_data_list[i])
        all_indices = extra_indices + main_indices
        random.shuffle(all_indices)
        return all_indices

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


def spilt_dataset(dataset, r: float, tx = None, ty = None):
    sze = len(dataset)
    shuffled_indices = np.random.permutation(sze)
    datasets_after_spilt = []
    datasetX = CustomSubset(dataset, shuffled_indices[0: int(sze * r)], tx)
    datasetY = CustomSubset(dataset, shuffled_indices[int(sze * r):], ty)
    datasets_after_spilt.extend([datasetX, datasetY])

    return datasets_after_spilt


# def process_dataset(dataset: Dataset, train_ratio, abs_ratio, train_transform = None , test_transform = None, aug_transform = None):
#     #划分训练集和测试集, 并施以不同的transform
#     train_data, eval_data = spilt_dataset(dataset, train_ratio, train_transform, test_transform)
#     #train_data, test_data = spilt_dataset(dataset , train_ratio, tx , ty)
#     # 进一步将训练集划分为绝对信息和相对信息训练集
#     abs_train_dataset, rel_train_dataset = spilt_dataset(train_data, abs_ratio, )
#     test_data, valid_data = spilt_dataset(eval_data, 0.5)
#     #return abs_train_dataset,rel_train_dataset, test_data, valid_data
#     return {
#         'ab_train' : abs_train_dataset,
#         're_train' : rel_train_dataset,
#         'test' : test_data,
#         'valid' : valid_data
#     }

def process_dataset(dataset: Dataset, train_ratio, abs_ratio, train_transform = None , test_transform = None, aug_transform = None):
    sze = len(dataset)
    shuffled_indices = np.random.permutation(sze)
    # train dataset
    train_sze = ceil(sze * train_ratio)
    ab_indices = range(ceil(train_sze * abs_ratio))
    re_indices = range(ceil(train_sze * abs_ratio) , train_sze)

    test_sze = (sze - train_sze) // 2
    test_indices = range(train_sze ,  train_sze + test_sze)
    val_indices = range(train_sze + test_sze , sze)

    abs_train_dataset = CustomSubset(dataset, shuffled_indices[ab_indices], train_transform)
    rel_train_dataset = CustomSubset(dataset, shuffled_indices[re_indices], aug_transform if aug_transform else train_transform)
    test_data = CustomSubset(dataset, shuffled_indices[test_indices], test_transform)
    valid_data = CustomSubset(dataset, shuffled_indices[val_indices], test_transform)

    return {
        'ab_train' : abs_train_dataset,
        're_train' : rel_train_dataset,
        'test' : test_data,
        'valid' : valid_data
    }

def match_partial_pairs(batch):
    # 按照标签分类
    # [image , label]
    classes = []
    cls2ind = {}
    data_seq = []
    for idx, (data, label) in enumerate(batch):
        data_seq.append(data)
        if cls2ind.get(label) is None:
            cls2ind[label] = []
            classes.append(label)
        cls2ind[label].append(idx)

    n = len(classes)
    partial_pairs_indices , labels = [], [],

    for i in range(n - 1):
        for j in range(i + 1, n):
            a = 1 if classes[i] > classes[j] else 0
            b = a ^ 1
            indicesx, indicesy = cls2ind[classes[i]], cls2ind[classes[j]]
            for (x, y) in zip(indicesx, indicesy):
                partial_pairs_indices.extend([[x , y],[y , x]])
                labels.extend([a, b])
    partial_pairs_indices = torch.tensor(partial_pairs_indices)
    data = torch.stack(data_seq)
    labels = torch.tensor(labels)
    return data, partial_pairs_indices, labels

def construct_partial_pairs(labels ,device):
    labels = labels.tolist()
    cls2ind = defaultdict(list)
    for idx, label in enumerate(labels):
        cls2ind[label].append(idx)
    new_labels = []
    partial_pairs_indices = []
    for ci in cls2ind.keys():
        for cj in cls2ind.keys():
            if ci == cj:
                continue
            label = 1 if ci > cj else 0
            indicesx, indicesy = cls2ind[ci], cls2ind[cj]
            for (x, y) in zip(indicesx, indicesy):
                partial_pairs_indices.append([x, y])
                new_labels.append(label)
    partial_pairs_indices = torch.tensor(partial_pairs_indices, device = device)
    labels = torch.tensor(new_labels, device = device)
    # 打乱一下顺序, shuffle
    # shuffle_idx = torch.randperm(labels.size(0))
    # return partial_pairs_indices[shuffle_idx], labels[shuffle_idx]
    return partial_pairs_indices , labels

def online_match(labels : torch.Tensor, r):
    N = len(labels)
    M = int(N * r)
    pre_idx , succ_idx = [] , []
    new_labels = []
    for i in range(N - 1):
        for j in range(i + 1 , N):
            if labels[i] == labels[j]: continue
            a = 1 if labels[i] > labels[j] else 0
            pre_idx.extend([i , j])
            succ_idx.extend([j , i])
            new_labels.extend([a , a ^ 1])
            if len(new_labels) > M:
                break
    L = min(M , len(new_labels))
    return np.array(pre_idx[ : L]) , np.array(succ_idx[ : L]), torch.tensor(new_labels[ : L])

def loader_init(args , abs_train_dataset , rel_train_dataset, test_dataset, val_dataset):
    ######################################## Absolute information DataLoader #################################
    samplerA = MPerClassSampler(
        abs_train_dataset.labels,
        batch_size=args.batch_size,
        m=args.M,
        iter_per_epoch=(len(abs_train_dataset) + args.batch_size - 1) // args.batch_size
    )
    abs_train_loader = DataLoader(
        abs_train_dataset,
        batch_sampler=samplerA,
        pin_memory=True,
        num_workers=args.workers
    )
    #absolute informat evaluate dataloader  -------fix the sequence --------a copy
    abs_eval_loader = DataLoader(
        abs_train_dataset,
        shuffle = False,
        batch_size = args.eval_batch_size,
        pin_memory = True,
        num_workers = args.workers
    )
    ######################################## Relative information DataLoader #################################
    # P = len(dataset.classes) if args.P is None else args.P
    P = args.P
    batch_size = P * args.K
    iter_per_epoch = (len(rel_train_dataset) + batch_size - 1) // batch_size if args.iter_per_epoch is None else args.iter_per_epoch
    samplerR = MPerClassSampler(
        rel_train_dataset.labels,
        batch_size=batch_size,
        m=args.K,
        iter_per_epoch=iter_per_epoch
    )

    rel_train_loader = DataLoader(
        rel_train_dataset,
        pin_memory=True,
        batch_sampler=samplerR,
        num_workers=args.workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return abs_train_loader , abs_eval_loader, rel_train_loader, test_loader, val_loader

def details_info_print(datasets , classes):
    print('=' * 20 + " The distributions of datasets " + "=" * 20)
    header = ["Class"] + classes + ["Total"]
    n , m = len(classes) , len(datasets)
    # data -> (n + 1) * len(datasets)
    data = [header]
    for name , dataset in datasets.items():
        unique_labels , counts = np.unique(dataset.labels, return_counts = True)
        counts_dict  = dict(zip(unique_labels, counts))
        #print(counts_dict)
        data.append([name] + [counts_dict[cls] if cls in counts_dict else 0 for cls in classes] + [len(dataset.labels)])

    # computer the max width of every column
    col_widths = [max(len(str(row[i])) for row in data) for i in range(len(header))]
    # print data
    s = False
    for row in data:
        print(' | '.join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
        if not s:
            print('-' * (sum(col_widths) + 3 * len(header) - 3)) # print separator
            s = True



if __name__ == "__main__":
    # 用法
    Datasets = [] # Dataset List
    datasets = MultiTaskDataset(Datasets)
    Sampler = MTBatchSampler(Datasets , batch_size = 64)
    task_seq = Sampler.getTaskSeq()
    dataloader = DataLoader(
        datasets,
        batch_sampler = Sampler,
        pin_memory = True,
    )
