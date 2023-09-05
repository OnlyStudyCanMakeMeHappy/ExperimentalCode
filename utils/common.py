import os
from typing import TextIO
import numpy
import torch
import torch.nn as nn
from torchvision import transforms
import tqdm
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import logging, colorlog
import faiss
from torch.utils.data import DataLoader
import time

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # CUDA采用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化机制

def freeze_BN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
def timer(func):
    def wrapper(*args , **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Evaluate time cost :{end_time - start_time :.2f}")
        return result
    return wrapper


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

    s = 1
    # 亮度、对比度、饱和度和色调
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )

    aug_transform = transforms.Compose([
        #transforms.RandomResizedCrop(224), # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.GaussianBlur(kernel_size=5),
        #transforms.RandomRotation(15),
        #transforms.RandomPerspective(distortion_scale=0.25, p=0.8),
        #transforms.RandomApply([color_jitter], p=0.8), # 以0.8的概率进行颜色抖动
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return base_transform, aug_transform, test_transform

def dict2str(d , s = ':'):
    return "".join([f"{k}{s}{v}," for k,v in d.items()])[:-1]   

def runtime_env(args , **kwargs):
    base_hp = {
        "batch_size":args.batch_size,
        "lr":args.lr,
        "epochs":args.epochs,
        'delta' : args.delta,
        'lr_schdeuler' : args.ls
    }    
    if args.fuse:
        base_hp["mu"] = args.mu
        base_hp["lambda"] = args.Lambda
        base_hp["vartheta"] = args.vartheta
        base_hp["varepsilon"] = args.varepsilon
    for k , v in kwargs.items():
        base_hp[k] = v
    return base_hp


def record(hyper_params, metrics):
    with open("logs.txt" , 'a') as f:
        #f.write(dict2str(hyper_params) + '\n')
        f.write(str(hyper_params)+ '\n')
        if isinstance(metrics , dict):
            f.write(dict2str(metrics , "=") + '\n')
        elif isinstance(metrics , str):
            f.write(metrics)
        f.write('\n---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  --------------- \n-:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::-  -:::::::::::::- \n---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------\n')

# def KNN_ind(reference_embeddings, reference_labels, query_embeddings, k):
#     #test在reference中找topk
#     #small query batch, small index: CPU is typically faster
#     dim = reference_embeddings.size(1)
#     index = faiss.IndexFlatL2(dim)
#     index.add(reference_embeddings.numpy())
#     # query与自身距离最近, 找 k + 1近邻, 然后ignore自身
#     D, I = index.search(query_embeddings.numpy() , k + 1)
#     # I是一个 query_size * k 的二维下标
#     # 四舍五入将浮点数转换为整数
#     return reference_labels.numpy()[I[ : , 1 : ]]
def KNN_ind(reference_embeddings, reference_labels, query_embeddings, k, metric = "Cosine"):
    if metric == 'Cosine':
    # 计算距离矩阵, 进行了L2 normalize, 直接计算点积即可
        dist_matrix = torch.mm(query_embeddings, reference_embeddings.T)
    # dim = 1 , 沿着dim = 1的方向, 计算每一行的topk
    elif metric == 'L2':
        #cdist doesn't work for float16
        if reference_embeddings.dtype == torch.float16:
            raise Exception("The tensor type is torch.float16 which is not support of cdist ")
        dist_matrix = -torch.cdist(query_embeddings, reference_embeddings)
    _ , knn_indices = torch.topk(dist_matrix , k , dim = 1)
    return reference_labels[knn_indices].numpy()

def predict(reference_embeddings, reference_labels,test_embeddings, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(reference_embeddings.cpu().numpy(), reference_labels.cpu().numpy())
    pred = neigh.predict(test_embeddings.cpu().numpy())
    return pred

def compute_c_index(labels : numpy.ndarray, predict):
    n = labels.shape[0]
    cnt = 0
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                cnt += 1
                # predict[i] == predict[j] and labels[i] > labels[j] x
                s += (predict[i] == predict[j]) / 2 + ((predict[i] < predict[j]) and (labels[i] < labels[j]) or (predict[i] > predict[j]) and (labels[i] > labels[j]))
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

@timer
#def Evaluation(test_dataset, train_dataset, model ,args , k = 10):
def Evaluation(test_loader, train_loader, model ,args):
#def Evaluation(test_loader, proxies, model ,args):
    reference_embeddings, reference_labels = get_embeddings_labels(train_loader, model, args)
    test_embeddings, test_labels = get_embeddings_labels(test_loader, model, args)

    # knn_indices = KNN_ind(reference_embeddings, reference_labels, test_embeddings, args.k)
    # pred = np.round(np.mean(knn_indices, axis=1))

    pred = predict(reference_embeddings, reference_labels, test_embeddings, args.k)
    #test_labels = test_labels.numpy()
    test_labels = test_labels.cpu().numpy()
    acc = np.mean(pred == test_labels)

    return {
    "ACC" : acc,
    "MAE" : mean_absolute_error(pred , test_labels),
    "MSE" : mean_squared_error(pred , test_labels),
    "QWK" : cohen_kappa_score(test_labels, pred , weights='quadratic'),
    "C_index" : compute_c_index(test_labels, pred)
}
def Evaluation_P(test_loader, proxies, model ,args):
    test_embeddings, test_labels = get_embeddings_labels(test_loader, model, args)
    with torch.no_grad():
        P = torch.nn.functional.normalize(proxies , dim = 1).to(args.gpu)
        _, nn_idx = torch.topk(-torch.cdist(test_embeddings , P), k=1)
        pred = nn_idx.squeeze().cpu().numpy()

    test_labels = test_labels.cpu().numpy()
    acc = np.mean(pred == test_labels)
    return {
        "ACC": acc,
        "MAE": mean_absolute_error(pred, test_labels),
        "MSE": mean_squared_error(pred, test_labels),
        "QWK": cohen_kappa_score(test_labels, pred, weights='quadratic'),
        "C_index": compute_c_index(test_labels, pred)
    }

def get_embeddings_labels(data_loader, model, args):
    model.eval()
    device = 'cpu' if args.gpu is None else args.gpu
    embeddings = torch.Tensor().to(device)
    labels = torch.LongTensor().to(device)
    with torch.no_grad():
        for (input, target) in data_loader:
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(input)
            #embeddings = torch.cat((embeddings, output.cpu()), 0)
            embeddings = torch.cat((embeddings, output), 0)
            labels = torch.cat((labels, target))
    return embeddings, labels







