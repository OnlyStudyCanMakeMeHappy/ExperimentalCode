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

def runtime_env(args , **kwargs):
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
    for k , v in kwargs.items():
        base_hp[k] = v
    return base_hp


def record(hyper_params, metrics):
    with open("logs.txt" , 'a') as f:
        f.write(dict2str(hyper_params) + '\n')
        f.write(dict2str(metrics , "=") + '\n')        
        #f.write('\n===***===***===***===***===***===***===\n/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\n===***===***===***===***===***===***===\n')
        f.write("\n.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.\n|=.--. |=.--. |=.--. |=.--. |=.--. |D.--. |E.--. |L.--. |I.--. |M.--. |I.--. |T.--. |E.--. |R.--. |=.--. |=.--. |=.--. |=.--. |=.--. |\n| (\\/) | (\\/) | (\\/) | (\\/) | (\\/) | :/\\: | (\\/) | :/\\: | (\\/) | (\\/) | (\\/) | :/\\: | (\\/) | :(): | (\\/) | (\\/) | (\\/) | (\\/) | (\\/) |\n| :\\/: | :\\/: | :\\/: | :\\/: | :\\/: | (__) | :\\/: | (__) | :\\/: | :\\/: | :\\/: | (__) | :\\/: | ()() | :\\/: | :\\/: | :\\/: | :\\/: | :\\/: |\n| '--'=| '--'=| '--'=| '--'=| '--'=| '--'D| '--'E| '--'L| '--'I| '--'M| '--'I| '--'T| '--'E| '--'R| '--'=| '--'=| '--'=| '--'=| '--'=|\n`------`------`------`------`------`------`------`------`------`------`------`------`------`------`------`------`------`------`------'\n")
def KNN_ind(reference_embeddings, reference_labels, query_embeddings, k):
    #test在reference中找topk
    #small query batch, small index: CPU is typically faster
    dim = reference_embeddings.size(1)
    index = faiss.IndexFlatL2(dim)
    index.add(reference_embeddings.numpy())
    # query与自身距离最近, 找 k + 1近邻, 然后ignore自身
    D, I = index.search(query_embeddings.numpy() , k + 1)
    # I是一个 query_size * k 的二维下标
    # 四舍五入将浮点数转换为整数
    return reference_labels.numpy()[I[ : , 1 : ]]


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

def Evaluation(test_loader, train_loader, model, device = None ,k = 5):
    reference_embeddings, reference_labels = get_embeddings_labels(train_loader, model, device)
    test_embeddings, test_labels = get_embeddings_labels(test_loader, model, device)
    knn_indices = KNN_ind(reference_embeddings, reference_labels, test_embeddings, k)
    pred = np.round(np.mean(knn_indices, axis=1))
    test_labels = test_labels.numpy()
    acc = np.mean(pred == test_labels)
    return {
    "ACC" : acc,
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






