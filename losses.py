import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from pytorch_metric_learning.losses import TripletMarginLoss,MarginLoss, MultiSimilarityLoss
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from collections import defaultdict

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

def Tripletloss(args):
    dist_pair = DPair()
    # default -> AvgNonZeroReducer
    loss_funcR = TripletMarginLoss(distance=dist_pair, margin = args.vartheta, reducer = MeanReducer())
    loss_funcA = TripletMarginLoss(margin = args.delta, reducer = MeanReducer())
    #return loss_funcR, loss_funcA
    return loss_funcR, AbsPartLoss(args.delta)

class AbsPartLoss(torch.nn.Module):
    def __init__(self, margin):
        super(AbsPartLoss, self).__init__()
        self.loss_funcA = TripletMarginLoss(margin = margin, reducer = MeanReducer())
        self.margin = margin
    def forward(self, x, Y):
        grouped_data = defaultdict(list)
        for i, label in enumerate(Y):
            grouped_data[label.item()].append(i)
        # tensor nonhash
        grouped_data = {k: torch.LongTensor(v) for k, v in grouped_data.items()}
        a_indices = b_indices = c_indices = torch.LongTensor()
        for (a, b, c) in torch.combinations(torch.unique(Y, sorted=True), 3):
            grid_a, grid_b, grid_c = torch.meshgrid(
                grouped_data[a.item()],
                grouped_data[b.item()],
                grouped_data[c.item()],
                indexing="ij"
            )
            a_indices = torch.cat([a_indices, grid_a.flatten()])
            b_indices = torch.cat([b_indices, grid_b.flatten()])
            c_indices = torch.cat([c_indices, grid_c.flatten()])
        dist = torch.cdist(x , x)
        loss1 = self.loss_funcA(x , Y)
        loss2 = torch.mean(F.relu(self.margin - dist[a_indices, c_indices] + dist[a_indices, b_indices]) + F.relu(
            self.margin - dist[c_indices, a_indices] + dist[c_indices, b_indices]))
        return loss1 + loss2



class ProxyNCALoss(torch.nn.Module):
    def __init__(self, num_classes, dim, device):
        super(ProxyNCALoss , self).__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, dim))
#        torch.nn.init.kaiming_normal_(self.proxies)
        torch.nn.init.xavier_uniform_(self.proxies)
        self.classes = torch.arange(num_classes)


    def forward(self, x, y : torch.Tensor):
        device = x.device
        self.proxies.data = self.proxies.data.to(device)
        self.classes = self.classes.to(device)
        P = F.normalize(self.proxies, p = 2, dim = -1)
        D = -torch.cdist(x , P)
        prob = F.softmax(D, dim = 1)
        exp = torch.sum(prob * (y.unsqueeze(1) == self.classes) , dim = 1)
        loss = torch.mean(-torch.log(exp) / (1-exp))
        return loss


class ProxyRankingLoss(torch.nn.Module):
    def __init__(self, num_classes, dim, margin = 0.1):
        super(ProxyRankingLoss, self).__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, dim))
        self.classes = torch.arange(num_classes)
        self.margin = margin
        torch.nn.init.xavier_uniform_(self.proxies)

    # Cross-Entropy , Unimodal
    def forward(self, x, y):
        device = x.device
        self.proxies.data = self.proxies.data.to(device)
        self.classes = self.classes.to(device)
        P = F.normalize(self.proxies, p=2, dim=-1)
        D = torch.cdist(x, P)
        sim = -torch.log(1 + D ** 2)
        prob = F.softmax(sim, dim=1)
        mask_e = y.unsqueeze(1) == self.classes
        exp = torch.sum(prob * mask_e, dim=1)  # broadcast
        loss1 = torch.mean(-torch.log(exp) / (1 - exp))
        #loss1 = torch.mean(-torch.log(exp))

        n = len(self.classes)
        # loss2 = 0.0
        # for i, label in enumerate(y):
        #     if label >= 2:
        #         indices = torch.combinations(self.classes[:label])
        #         p , s = indices[:,0] , indices[:,1]
        #         loss2 += torch.mean(F.relu(self.margin - D[i][p] + D[i][s]))
        #     if  n - label - 1 >= 2:
        #         indices = torch.combinations(self.classes[label + 1:])
        #         p , s = indices[:,0] , indices[:,1]
        #         loss2 += torch.mean(F.relu(self.margin - D[i][s] + D[i][p]))

        # 代理排序
        # D = torch.cdist(P , P)
        # diff = D[:, : -1] - D[:, 1:]
        # mask = torch.where(self.classes.unsqueeze(1) > self.classes, 1, -1)
        # loss2 = torch.mean(F.relu(self.margin - mask[:, : -1] * diff))

        #D = torch.matmul(P , P.T)
        # y_i , (0 , y_i - 1) & (y_i + 1 , n - 1)
        # 只考虑邻接类别
        diff = D[:, : -1] - D[:, 1 : ]
        #mask = torch.where(y.unsqueeze(1) > self.classes, 1, -1)
        mask = torch.where(y.unsqueeze(1) >= self.classes, 1, -1)
        #loss2 = torch.mean(F.relu(self.margin - mask[:, : -1] * diff))
        loss2 = torch.mean(F.relu(0 - mask[:, : -1] * diff))
        #loss2 = torch.mean(F.softplus(self.margin - mask[:, : -1] * diff))
        #loss2 = torch.mean(torch.log(1 + torch.sum(torch.exp(self.margin - mask[:, : -1] * diff), dim = 1)))
        #loss2 = torch.mean(F.softplus(torch.logsumexp(self.margin - mask[:, : -1] * diff, dim = 1)))
        return loss1 + loss2

        #return loss2

class SemicircularProxiesLearner(nn.Module):
    def __init__(self, num_ranks, dim):
        super(SemicircularProxiesLearner, self).__init__()

        self.num_ranks = num_ranks
        self.rank_ids = nn.Parameter(torch.arange(num_ranks)[:, None].float(), requires_grad=False)

        self.v0 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        self.v1 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.v0)
        nn.init.xavier_normal_(self.v1)

    def forward(self):
        theta = self.rank_ids * np.pi / (self.num_ranks - 1)
        gamma = torch.cosine_similarity(self.v0, self.v1).arccos()
        norm_v0 = self.v0 / torch.linalg.norm(self.v0, dim=-1)
        norm_v1 = self.v1 / torch.linalg.norm(self.v1, dim=-1)
        proxies = (gamma - theta).sin() / gamma.sin() * norm_v0 + theta.sin() / gamma.sin() * norm_v1
        return proxies
class HardCplLoss(nn.Module):
    @staticmethod
    def forward(X, gt, proxies):
        # [batch_size , dim] , [N, dim]
        assign_metric = torch.cosine_similarity(X[: , None , :] , proxies[None , : , :], dim = -1)
        proxies_metric = torch.cosine_similarity(proxies , proxies, dim = -1)
        selected_proxies_metric = proxies_metric[gt, :].detach()  # [B, C]
        loss = F.kl_div(F.log_softmax(assign_metric, dim=-1), F.softmax(selected_proxies_metric, dim=-1), reduction='batchmean')
        return loss

if __name__ == "__main__":
    embedding = torch.randn(10, 64)
    num_classes = 10
    labels = torch.randint(0 , num_classes, (10,))
    criterion = ProxyNCALoss(num_classes, 64)
    criterion(embedding , labels)
    print(criterion.get_proxies())
