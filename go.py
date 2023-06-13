import argparse
import re
import time
import os
import torch.nn.functional
from model.embedding import Embedding
from model.mlp import MLP
import model.backbone as backbone
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
import glob
import pandas as pd
from utils.common import *
from pytorch_metric_learning.losses import TripletMarginLoss,MarginLoss, MultiSimilarityLoss
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from MPerClassSampler import MPerClassSampler
from torch.utils.tensorboard import SummaryWriter

# 启动命令 : tensorboard --logdir=/path/to/logs/ --port=xxxx
parser = argparse.ArgumentParser(description="Train Model")

# worker -> windows : 0
parser.add_argument('--data', default='/home/tuijiansuanfa/users/cjx/data/FGNET/images', type=str, help='path of dataset')
parser.add_argument('--dim', default=64, type=int, help='embedding size')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('-M', default = 6, type = int , help = 'MPerClassSampler')
#==============================batch size==========================#
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_batch_size', default=120, type=int)
#==============================learning rate==========================#
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)


parser.add_argument('--gpu', default=None, type=int)
#==============================hyper parameters of relative information==============================#
parser.add_argument('--fuse' , action = "store_true", help = "whether fuse the relative information")
parser.add_argument('--loss', default='triplet', choices=['ms', 'triplet', 'margin'])
parser.add_argument('--mu', default = 0.8,type = float)
parser.add_argument('--Lambda', default = 0.8,type = float)
# 绝对信息的margin
parser.add_argument('--delta', default= 0.5,type = float)
parser.add_argument('--vartheta', '-vt', default = 0.5,type = float)
parser.add_argument('--varepsilon', '-ve', default=0.5, type = float)

args = parser.parse_args()

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
class DPair(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        # query_emb.size() = (batch_size , dimension)
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        # 返回一个二维网格, rows每一行元素都是相同, cols每一列元素相同
        rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
        output = torch.zeros(rows.size(), dtype=dtype, device=device)
        rows, cols = rows.flatten(), cols.flatten()
        # rows.size() = cols.size() = query_emb.size(0) * ref_emb.size(0)
        distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
        output[rows, cols] = distances
        return output

    def pairwise_distance(self, query_emb, ref_emb):
        # (batch_size ** 2, dim * 2)
        N = query_emb.size(1) // 2
        return torch.nn.functional.pairwise_distance(query_emb[:, : N], ref_emb[:, : N]) \
            + torch.nn.functional.pairwise_distance(query_emb[:, N :], ref_emb[:, N:])

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

    def match_pair(slef , data: list[Dataset]) -> list:
        res = []
        cmp = lambda x, y: 1 if x > y else 0
        # data是一个二维列表, 实现(data[i] , data[j])两两配对
        for i in range(len(data) - 1):
            for j in range(i + 1, len(data)):
                res.extend([(x[0], y[0], cmp(x[1], y[1])) for (x, y) in zip(data[i], data[j]) if x[1] != y[1]])
                res.extend([(x[0], y[0], cmp(x[1], y[1])) for (y, x) in zip(data[i], data[j]) if x[1] != y[1]])
        return res
class FGNETDataset(Dataset):
    def __init__(self, root, transform = None):
        file_path = root
        file_names = os.listdir(root)
        img_paths = [os.path.join(file_path, file_name) for file_name in file_names]
        self.images = [Image.open(img_path).convert('RGB') for img_path in img_paths]
        self.labels = [int(re.match(r'\d{3}A(\d+)\w?.JPG', name).group(1)) for name in file_names]
        self.transform = transform
        intervals = [0, 3, 11, 16, 24, 40]
        self.groupIds = list()
        for label in self.labels:
            for i in range(len(intervals) - 1, -1, -1):
                if label >= intervals[i]:
                    self.groupIds.append(i)
                    break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        #label = self.labels[index]
        label = self.groupIds[index]
        return image, label
class AdienceDataset(Dataset):
    def __init__(self, root, transform=None):
        # 获取每张图像的绝对路径
        # self.img_path = os.path.join(root , 'images')
        age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), ('(8, 12)', '8-13'),
                       ('13', '8-13'), ('22', '15-20'), ('(8, 23)', '15-20'), ('23', '25-32'), ('(15, 20)', '15-20'),
                       ('(25, 32)', '25-32'), ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'),
                       ('(38, 42)', '38-43'), ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), ('45', '38-43'),
                       ('(38, 43)', '38-43'), ('(38, 42)', '38-43'), ('(38, 48)', '48-53'), ('46', '48-53'),
                       ('(48, 53)', '48-53'), ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'), ('57', '60+'),
                       ('58', '60+')]
        age_mapping_dict = dict(age_mapping)
        age_to_label_map = {
            '0-2': 0,
            '4-6': 1,
            '8-13': 2,
            '15-20': 3,
            '25-32': 4,
            '38-43': 5,
            '48-53': 6,
            '60+': 7
        }
        anna_file_path = glob.glob(root + "/targets/*.txt")
        data_list = [pd.read_csv(path, delimiter='\t', ) for path in anna_file_path]
        data = pd.concat(data_list, ignore_index=True)
        self.labels = []
        self.image_path = []
        self.transform = transform
        for _, row in data.iterrows():
            if row.age != 'None':
                self.labels.append(age_to_label_map[age_mapping_dict[row.age]])
                img_path = f"{root}/images/{row.user_id}/landmark_aligned_face.{row.face_id}.{row.original_image}"
                self.image_path.append(img_path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def train(
        Backbone,
        Rel_Net,
        Abs_Net,
        loss_funcA,
        loss_funcB,
        pop_train_loader,
        abs_train_loader ,
        aug_transform,
        optimizer,
        epoch,
        writer,
        logger,
):
    start_time = time.time()
    Backbone.train()
    freeze_BN(Backbone)
    lossAList, lossRList = [] , []
    if args.fuse:
        for idx,(x, y, label) in enumerate(pop_train_loader):
            x = x.to(args.gpu, non_blocking = True)
            y = y.to(args.gpu, non_blocking = True)
            label = label.to(args.gpu)
            x_embedding,y_embedding = Rel_Net(x) ,Rel_Net(y)
            loss1 = loss_funcA(torch.cat((x_embedding, y_embedding), dim = 1), label)
            aug_x = aug_transform(x)
            aug_embedding = Rel_Net(aug_x)
            loss2 = torch.mean(
                torch.nn.functional.relu(
                    torch.nn.functional.pairwise_distance(x_embedding, aug_embedding) -
                    torch.nn.functional.pairwise_distance(y_embedding, aug_embedding) +
                    args.varepsilon
                )
            )
            loss = loss1 + args.mu * loss2
            #loss = loss1
            lossRList.append(loss.item())
            loss *= args.Lambda
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for idx,(image , label) in enumerate(abs_train_loader):
        image = image.to(args.gpu, non_blocking = True)
        label = label.to(args.gpu, non_blocking = True)
        embedding = Abs_Net(image)
        loss = loss_funcB(embedding, label)
        lossAList.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    lossA , lossR = np.mean(lossAList) , np.mean(lossRList)
    loss_total = lossA + args.Lambda * lossR

    if args.fuse:
        writer.add_scalars("loss" , {
            "lossA" : lossA,
            "lossR": lossR,
            "loss_toal": loss_total,
        })
        logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , lossR : {lossR}, loss_total : {loss_total} , time_cost : {end_time - start_time : .2f}s")
    else:
        writer.add_scalar('only_absolute_information', lossA, epoch)
        logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , time_cost = {end_time - start_time:.2f}s")


def main():
    fix_seed(0)
    assert args.gpu is not None, "GPU is necessary"
    Backbone = backbone.ResNet50().to(args.gpu)
    Abs_Head = Embedding(Backbone.output_size, embedding_size=args.dim).to(args.gpu)
    Rel_Head = MLP(Backbone.output_size, embedding_size=args.dim, hidden_size=512).to(args.gpu)
    Abs_Net = nn.Sequential(Backbone, Abs_Head)
    Rel_Net = nn.Sequential(Backbone, Rel_Head)
    #architecture.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))

    base_transform, aug_transform, test_transform = get_transforms()

    # 将数据集划分为K等分, 训练集:测试集 = K - 1 : 1, 进一步按照比例将训练划分为相对信息和绝对信息
    # random_spilt方法会丢失Dataset的labels等属性 , 所以继承SubSet自定义数据集划分
    # 数据集打乱 np.random.permutation
    # 训练数据和测试数据采用不同的transform
    def spilt_dataset(dataset , r : float, tx = None , ty = None):
        sze = len(dataset)
        shuffled_indices = np.random.permutation(sze)
        datasetX = CustomSubset(dataset, shuffled_indices[0: int(sze * r)], tx)
        datasetY = CustomSubset(dataset, shuffled_indices[int(sze * r) : ], ty)
        return datasetX , datasetY

    def process_dataset(dataset: Dataset, K: int = 10, N: int = 10, abs_ratio: float = 0.1):
        train_data , test_data = spilt_dataset(dataset, (K - 1) / K, base_transform, test_transform)
        abs_train_dataset , rel_train_dataset = spilt_dataset(train_data, abs_ratio)
        partial_order_pair = POP(rel_train_dataset, N)
        return (abs_train_dataset, partial_order_pair, test_data)

    dataset = FGNETDataset(root = args.data)
    abs_train_dataset, pop_dataset, test_dataset = process_dataset(dataset)
    sampler = MPerClassSampler(
        abs_train_dataset.labels,
        batch_size = args.batch_size,
        m = args.M,
        iter_per_epoch = len(abs_train_dataset) // args.batch_size
    )
    # M * (N - 1) // N * K
    # M * (N - 1) // N * (1 - K)
    abs_train_loader = DataLoader(
        abs_train_dataset,
        batch_size = args.batch_size,
        pin_memory = True,
        sampler = sampler,
        num_workers = args.workers
    )
    pop_train_loader = DataLoader(
        pop_dataset,
        batch_size = args.batch_size,
        pin_memory = True,
        shuffle = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.eval_batch_size,
    )

    params_group = [
        {'params': Backbone.parameters(), 'lr': args.lr},
        {'params': Abs_Head.parameters(), 'lr': args.lr * 10},
        {'params': Rel_Head.parameters(), 'lr': args.lr * 10}
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)
    dist_pair = DPair()
    loss_funcA = TripletMarginLoss(distance=dist_pair, margin = args.vartheta)
    loss_funcB = TripletMarginLoss(margin = args.delta)
    hparams = runtime_env(args)
    writer = SummaryWriter(os.path.join("runs" , dict2str(hparams)))
    # getlogger
    logger = get_logger()
    print("==============START TRAING================")
    for epoch in range(1 , args.epochs + 1):
        train(Backbone, Rel_Net, Abs_Net, loss_funcA, loss_funcB, pop_train_loader, abs_train_loader , aug_transform, optimizer, epoch, writer , logger)
    eval_result = Evaluation(test_loader, abs_train_loader, model = Abs_Net, device = args.gpu)
    logger.info("MAE = {MAE} , MSE = {MSE} , QWK = {QWK}, C-index = {C_index}".format(**eval_result))
    record(hparams, eval_result)




if __name__ == "__main__":
    main()
