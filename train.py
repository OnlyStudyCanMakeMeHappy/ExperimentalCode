import argparse
import re
import torchvision.datasets as datasets
from model.embedding import Embedding
from model.mlp import MLP
import model.backbone as backbone
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import pandas as pd
import time
from losses import RankingLoss, TripleLoss
from utils.common import *
from pytorch_metric_learning.losses import MarginLoss, MultiSimilarityLoss
import cv2
from utils.MPerClassSampler import MPerClassSampler

parser = argparse.ArgumentParser(description="Train Model")

# worker -> windows : 0
parser.add_argument('--data', default='/home/chenjx/data/cars196', type=str, help='datasets path')
parser.add_argument('--dim', default=64, type=int, help='embedding size')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--eval_batch_size', default=120, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--P_task', default=0.8, type=float)
parser.add_argument('-gamma', default=0.8, type=float)
parser.add_argument('-N', default=4, type=int, help="The length of variance")
parser.add_argument('--save_dir', default='result', type=str)
parser.add_argument('--loss', default='triplet', choices=['ms', 'triplet', 'margin'])
parser.add_argument('--bs', action='store_true', help='whether use balanced sampler')
args = parser.parse_args()


# torch.set_num_threads(3)
class FGNETDataset(Dataset):
    def __init__(self, root, transform=None, N=4):
        self.file_path = root
        self.file_names = os.listdir(root)
        self.labels = [int(re.match(r'\d{3}A(\d+)\w?.JPG', name).group(1)) for name in self.file_names]
        intervals = [0, 3, 11, 16, 24, 40]
        self.groupIds = list()
        for label in self.labels:
            for i in range(len(intervals) - 1, -1, -1):
                if label >= intervals[i]:
                    self.groupIds.append(i)
                    break
        self.transform = transform
        self.N = N

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.file_path, self.file_names[index])
        # opencv图像处理比torchvision的transform快, 所以这里的base_transfrom只做ToTensor和通道顺序转换

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = self.transform(image)
        label = self.labels[index]
        # groupId = self.groupIds[index]
        # return images, label, groupId
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


def train(Backbone, DML_Head, SSL_Head, epoch, train_loader, DML_criterion, SSL_criterion, optimizer, aug_transform):
    loss_list = []
    optimizer_q = torch.optim.AdamW([
        {'params': Backbone.parameters(), 'lr': args.lr},
        {'params': DML_Head.parameters(), 'lr': args.lr * 10}],
        weight_decay=4e-4
    )
    optimizer_g = torch.optim.AdamW([
        {'params': Backbone.parameters(), 'lr': args.lr},
        {'params': SSL_Head.parameters(), 'lr': args.lr * 10}],
        weight_decay=4e-4
    )
    # Backbone.train()
    # for idx, (images, labels) in enumerate(train_loader):
    #     images = images.to(args.gpu, non_blocking=True)
    #     labels = labels.to(args.gpu, non_blocking=True)
    #     #print(labels)
    #     features = Backbone(images)
    #     embeddings = DML_Head(features)
    #     loss = DML_criterion(embeddings, labels)
    #     if np.random.uniform() < args.P_task:
    #         output = SSL_Head(features)
    #         aug_outputs = []
    #         for i in range(args.N):
    #             images = aug_transform(images)
    #             features = Backbone(images)
    #             aug_outputs.append(SSL_Head(features))
    #         #loss = loss + args.gamma * SSL_criterion(output, torch.stack(aug_outputs))
    #         loss = loss + SSL_criterion(output, torch.stack(aug_outputs))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     loss_list.append(loss.item())
    Backbone.train()
    freeze_BN(Backbone)
    for idx, (images, labels) in enumerate(train_loader):
        # 放在epoch里面
        total_loss = 0.0
        images = images.to(args.gpu, non_blocking=True)
        labels = labels.to(args.gpu, non_blocking=True)
        #print(labels)
        features = Backbone(images)
        embeddings = DML_Head(features)
        loss_metric = DML_criterion(embeddings, labels)
        total_loss += loss_metric.item()
        optimizer_q.zero_grad()
        #optimizer.zero_grad()
        #loss_metric.backward(retain_graph = True)
        loss_metric.backward()
        #optimizer.step()
        optimizer_q.step()


        if np.random.uniform() < args.P_task:
            features = Backbone(images)
            output = SSL_Head(features)
            aug_outputs = []
            for i in range(args.N):
                images = aug_transform(images)
                features = Backbone(images)
                aug_outputs.append(SSL_Head(features))
            loss_ranking =  SSL_criterion(output, torch.stack(aug_outputs))
            #loss_ranking = SSL_criterion(output, torch.stack(aug_outputs))
            total_loss += loss_ranking.item()
            optimizer_g.zero_grad()
            #optimizer.zero_grad()
            loss_ranking.backward()
            #optimizer.step()
            optimizer_g.step()

        loss_list.append(total_loss)
    return np.mean(loss_list)


def evaluation(net, loader, K=[1, 2, 4, 8]):
    net.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(args.gpu)
            embedding = net(images)
            embeddings.append(embedding)
            labels.append(targets)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        recall_k, MLRC = recall(embeddings, labels, K)
    for k, r in zip(K, recall_k):
        print(f"Recall@{k} = {r:.2f}, ", end="")
    print(f"MAP = {MLRC[0]}, RP = {MLRC[1]}")
    return recall_k, MLRC

def test(architecture, test_loader):
    print("-" * 20 + "Evaluating on test dataset" + "-" * 20)
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    architecture.load_state_dict(checkpoint)
    recall_k , MLRC = evaluation(nn.Sequential(architecture['backbone'], architecture['dml_head']), test_loader)
    Record(os.path.join(args.save_dir, 'evaluation_results.txt'), recall_k, MLRC)

def main():
    fix_seed(0)
    assert args.gpu is not None, "GPU is necessary"
    Backbone = backbone.ResNet50().to(args.gpu)
    # freeze BatchNorm2d
    # freeze_BN(Backbone)
    DML_Head = Embedding(Backbone.output_size, embedding_size=args.dim).to(args.gpu)
    SSL_Head = MLP(Backbone.output_size, embedding_size=args.dim, hidden_size=512).to(args.gpu)
    architecture = nn.ModuleDict({
        'backbone': Backbone,
        'dml_head': DML_Head,
        'ssl_head': SSL_Head,
    })
    #architecture.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))

    base_transform, aug_transform = get_transforms()
    train_dir = args.data + '/train'
    test_dir = args.data + '/test'
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=base_transform
    )
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
    )
    # TODO: BalancedSampler, dataset partitioning, optimizer
    sampler = MPerClassSampler(train_dataset, batch_size = args.batch_size, m = 4, iter_per_epoch=len(train_dataset) // args.batch_size)

    p = {'sampler' : None, 'shuffle' : True}
    if args.bs:
        p['sampler'] = sampler
        p['shuffle'] = None
    # ctrl + p : 参数信息
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=p['sampler'],
        shuffle=p['shuffle'],
        num_workers=args.workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=120,
        num_workers=args.workers,
    )
    # optimizer settings
    params_group = [
        {'params': Backbone.parameters(), 'lr': args.lr},
        {'params': DML_Head.parameters(), 'lr': args.lr * 10},
        {'params': SSL_Head.parameters(), 'lr': args.lr * 10}
    ]
    optimizer = torch.optim.AdamW(params_group, weight_decay=4e-4)

    if args.loss == 'triplet':
        DML_criterion = TripleLoss()
    elif args.loss == 'margin':
        DML_criterion = MarginLoss()
    elif args.loss == 'ms':
        DML_criterion = MultiSimilarityLoss()

    SSL_criterion = RankingLoss()

    print("-" * 20 + "start training" + "-" * 20)
    recalls_list = []
    losses_list = []
    best_recall = [0.]
    best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        loss = train(Backbone, DML_Head, SSL_Head, epoch, train_loader, DML_criterion, SSL_criterion, optimizer,
                     aug_transform)
        end_time = time.time()
        print(f"Epoch[{epoch + 1}/{args.epochs}]: loss = {loss}, time cost ={end_time - start_time:.2f}s ")

        val_recall, val_MLRC = evaluation(nn.Sequential(Backbone, DML_Head), train_eval_loader)
        recalls_list.append(val_recall)
        losses_list.append(loss)

        #if best_recall[0] < val_recall[0] or if best_recall[0] == val_recall[0] and best_recall[-1] < val_recall[-1]
        if best_recall[0] < val_recall[0] :
            best_recall = val_recall
            best_epoch = epoch
            best_MLRC = val_MLRC

            if args.save_dir is not None:
                torch.save(architecture.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                Record(os.path.join(args.save_dir, 'best_results.txt'), best_recall, best_MLRC, best_epoch)

    #----------------------Testing-----------------------#
    test(architecture,test_loader)



if __name__ == "__main__":
    main()
