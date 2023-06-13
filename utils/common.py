import os
from typing import TextIO
<<<<<<< HEAD
=======

>>>>>>> origin/main
import numpy
import torch
import torch.nn as nn
from torchvision import transforms
import tqdm
import random
import numpy as np
<<<<<<< HEAD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import logging, colorlog
=======
import faiss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
>>>>>>> origin/main
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def freeze_BN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# TODO: 根据模型做不同的transform, 特别是BnInception需要特别对待
def get_transforms(model: str = "ResNet50"):
    # opencv读取图片是BGR通道, ResNet : 转换通道, ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    """
        ImageNet预训练模型都建议Resize到(224,224)，使用该均值和方差normalize
        BNInception,(b,g,r)
    """
    base_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    aug_transform = transforms.Compose([
        transforms.transforms.RandomResizedCrop(224),
        transforms.RandomPerspective(distortion_scale=0.15, p=1),
        transforms.ColorJitter([1., 1.1], [1., 1.1], [1., 1.1]),
    ])
<<<<<<< HEAD
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    #return base_transform, transform, aug_transform
    return base_transform, aug_transform, test_transform

def dict2str(d , s = ':'):
    return "".join([f"{k}{s}{v}," for k,v in d.items()])[:-1]   

def runtime_env(args):
    base_hp = {
        "batch_size":args.batch_size,
        "lr":args.lr,
        "epochs":args.epochs,
        'delta' : args.delta
    }    
    if args.fuse:
        base_hp["mu"] = args.mu
        base_hp["lambda"] = args.Lambda
        base_hp["vartheta"] = args.vartheta
        base_hp["varepsilon"] = args.varepsilon
    return base_hp


def record(hyper_params, metrics):
    with open("logs.txt" , 'a') as f:
        f.write(dict2str(hyper_params) + '\n')
        f.write(dict2str(metrics , "=") + '\n')        
        f.write('\n===***===***===***===***===***===***===\n/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\n===***===***===***===***===***===***===\n')

def predict(reference_embeddings, reference_labels,test_embeddings, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(reference_embeddings.numpy(), reference_labels.numpy())
    pred = neigh.predict(test_embeddings.numpy())
    return pred

def compute_c_index(labels : numpy.ndarray, predict):
    n = labels.shape[0]
    cnt = 0
    s = 0.0
    for i in range(n):
        for j in range(i + 1):
            if labels[i] != labels[j]:
                cnt += 1
                s += (predict[i] == predict[j]) / 2 + (predict[i] < predict[j]) and (labels[i] < labels[j])
    return s / cnt

def get_logger(logger_name = None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # log_color有默认配置
    formatter = colorlog.ColoredFormatter(
        fmt = '%(date_log_color)s[%(asctime)s] %(level_log_color)s[%(levelname)s]: %(log_color)s%(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
        secondary_log_colors={
            'date' : {
                'INFO' : 'red',
            },
            'level' : {
                'INFO': 'blue',
            }
        },
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def Evaluation(test_loader, train_loader, model, device = None ,k = 5):
    reference_embeddings, reference_labels = get_embeddings_labels(train_loader, model, device)
    test_embeddings, test_labels = get_embeddings_labels(test_loader, model, device)
    pred = predict(reference_embeddings, reference_labels, test_embeddings, k)    
    test_labels = test_labels.numpy()
    acc = np.mean(pred == test_labels)
    return {
        "MAE" : mean_absolute_error(pred , test_labels),
        "MSE" : mean_squared_error(pred , test_labels),
        "QWK" : cohen_kappa_score(test_labels, pred),
        "C_index" : compute_c_index(test_labels, pred)
    }

def get_embeddings_labels(data_loader, model, device = None):
    model.eval()
    embeddings = torch.Tensor()
    labels = torch.LongTensor()
    with torch.no_grad():
        for (input, target) in (data_loader):
            if device is not None:
                input = input.cuda(device, non_blocking=True)
            output = model(input)
            embeddings = torch.cat((embeddings, output.cpu()), 0)
            labels = torch.cat((labels, target))
    return embeddings, labels


=======

    #return base_transform, transform, aug_transform
    return base_transform, aug_transform


def evaluation(embeddings, labels, K=[]):
    """
    内存有限,限定单次能计算的最大的N = 1e4
    :param args: 命令行参数
    :param embeddings: 嵌入
    :param labels: 标签
    :param K: K-NN
    :return:Recll@K
    """
    class_set, _, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)

    indices = []
    M = int(1e4)
    N = embeddings.size(0)
    assert N >= max(K), "batch size N mast >= max(K)"
    eval_iter = tqdm.tqdm(range((embeddings.size(0) + M - 1) // M), ncols=100)
    eval_iter.set_description("Recall Evaluation")
    for i in eval_iter:
        s = i * M
        e = min((i + 1) * M, N)

        Chunk = embeddings[s: e]
        sim_mat = torch.mm(Chunk, embeddings.t())
        # 经过了L2Norm => [-1:1]
        sim_mat[range(0, e - s), range(s, e)] = -2  # 将自相似度化为最小
        # values, indices : Tensor
        index = torch.topk(sim_mat, max(K))[1]  # return (values,indices)
        indices.append(index)

    indices = torch.cat(indices, dim=0)
    #
    topk_labels = labels[indices]
    # 标签按照相似度排序
    fmat = topk_labels == labels.unsqueeze(1)
    recall_k = []
    for k in K:
        acc = (fmat[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(acc)
    return recall_k
>>>>>>> origin/main


def recall(embeddings, labels, K=[]):
    knn_inds = []
    M = 10000
    class_set, inverse_indices, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    labels_counts = {k.item() : v for k, v in zip(class_set, counts)}


    evaluation_iter = tqdm.tqdm(range(embeddings.shape[0] // M + 1), ncols=80)
    evaluation_iter.set_description("Measuring recall...")
    for i in evaluation_iter:
        s = i * M
        e = min((i + 1) * M, embeddings.shape[0])
        # print(s, e)

        embeddings_select = embeddings[s:e]
        cos_sim = nn.functional.linear(embeddings_select, embeddings)
        cos_sim[range(0, e - s), range(s, e)] = 1e5
        knn_ind = cos_sim.topk(1 + max(max(counts), max(K)))[1][:, 1:]
        knn_inds.append(knn_ind)

    knn_inds = torch.cat(knn_inds, dim=0)

    selected_labels = labels[knn_inds]
    correct_labels = labels.unsqueeze(1) == selected_labels

    MLRC = (0, 0)

    mAP = []
    RP = []
    evaluation_iter = tqdm.tqdm(range(labels.shape[0]), ncols=80)
    evaluation_iter.set_description("Measuring MAP and RP")
    for i in evaluation_iter:

        cnt = counts[labels[i] - class_set[0]] - 1
        #cnt = labels_counts[labels[i].item()]
        l = correct_labels[i, 0:cnt].float()
        rp = l.sum() / cnt

        intersect_size = 0
        ap = 0
        for j in range(len(l)):

            if l[j]:
                intersect_size += 1
                precision = intersect_size / (j + 1)
                ap += precision / cnt

        RP.append(rp)
        mAP.append(ap)

    RP = sum(RP) / len(RP)
    mAP = sum(mAP) / len(mAP)

    MLRC = (mAP.item(), RP.item())

    recall_k = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)

    return recall_k, MLRC

<<<<<<< HEAD
def evaluation(embeddings, labels, K=[]):
    """
    内存有限,限定单次能计算的最大的N = 1e4
    :param args: 命令行参数
    :param embeddings: 嵌入
    :param labels: 标签
    :param K: K-NN
    :return:Recll@K
    """
    class_set, _, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)

    indices = []
    M = int(1e4)
    N = embeddings.size(0)
    assert N >= max(K), "batch size N mast >= max(K)"
    eval_iter = tqdm.tqdm(range((embeddings.size(0) + M - 1) // M), ncols=100)
    eval_iter.set_description("Recall Evaluation")
    for i in eval_iter:
        s = i * M
        e = min((i + 1) * M, N)

        Chunk = embeddings[s: e]
        sim_mat = torch.mm(Chunk, embeddings.t())
        # 经过了L2Norm => [-1:1]
        sim_mat[range(0, e - s), range(s, e)] = -2  # 将自相似度化为最小
        # values, indices : Tensor
        index = torch.topk(sim_mat, max(K))[1]  # return (values,indices)
        indices.append(index)

    indices = torch.cat(indices, dim=0)
    #
    topk_labels = labels[indices]
    # 标签按照相似度排序
    fmat = topk_labels == labels.unsqueeze(1)
    recall_k = []
    for k in K:
        acc = (fmat[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(acc)
    return recall_k
=======
def Record(file_path , recall_k, MLRC, epoch = None):
    with open(file_path, "w") as f:
        if epoch:
            f.write('Best Epoch: {}\n'.format(epoch))
        for i, K in enumerate([1, 2, 4, 8]):
            f.write("Recall@{}: {:.4f}\n".format(K, recall_k[i]))

        f.write("\nMAP@R: {:.4f}\n".format(MLRC[0]))
        f.write("RP: {:.4f}\n".format(MLRC[1]))

def predict(reference_embeddings, reference_labels,test_embeddings, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(reference_embeddings.numpy(), reference_labels.numpy())
    pred = neigh.predict(test_embeddings.numpy())
    return pred
def compute_c_index(labels : numpy.ndarray, predict):
    n = labels.shape[0]
    cnt = 0
    s = 0.0
    for i in range(n):
        for j in range(i + 1):
            if labels[i] != labels[j]:
                cnt += 1
                s += (predict[i] == predict[j]) / 2 + (predict[i] < predict[j]) and (labels[i] < labels[j])
    return s / cnt



def Evaluation(test_loader, train_loader, model, device = None ,k = 5):
    reference_embeddings, reference_labels = get_embeddings_labels(train_loader, model, device)
    test_embeddings, test_labels = get_embeddings_labels(test_loader, model, device)
    pred = predict(reference_embeddings, reference_labels, test_embeddings, k)
    test_labels = test_labels.numpy()
    return {
        "MAE" : mean_absolute_error(pred , test_labels),
        "MSE" : mean_squared_error(pred , test_labels),
        "QWK" : cohen_kappa_score(test_labels, pred),
        "C_index" : compute_c_index(test_labels, pred)
    }

def get_embeddings_labels(data_loader, model, device = None):
    model.eval()
    embeddings = torch.Tensor()
    labels = torch.LongTensor()
    with torch.no_grad():
        for (input, target) in (data_loader):
            if device is not None:
                input = input.cuda(device, non_blocking=True)
            output = model(input)
            embeddings = torch.cat((embeddings, output.cpu()), 0)
            labels = torch.cat((labels, target))
    return embeddings, labels




>>>>>>> origin/main
