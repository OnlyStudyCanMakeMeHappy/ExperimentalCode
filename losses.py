from pytorch_metric_learning.losses import TripletMarginLoss,MarginLoss, MultiSimilarityLoss
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch
import torch.nn.functional as F

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
    return loss_funcR, loss_funcA


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

    def forward(self, x, y):
        device = x.device
        self.proxies.data = self.proxies.data.to(device)
        self.classes = self.classes.to(device)
        P = F.normalize(self.proxies, p=2, dim=-1)
        D = torch.cdist(x, P)
        prob = F.softmax(-D, dim=1)
        mask_e = y.unsqueeze(1) == self.classes
        exp = torch.sum(prob * mask_e, dim=1)  # broadcast
        loss1 = torch.mean(-torch.log(exp) / (1 - exp))

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
        mask = torch.where(y.unsqueeze(1) > self.classes, 1, -1)
        loss2 = torch.mean(F.relu(self.margin - mask[:, : -1] * diff))
        #loss2 = torch.mean(F.softplus(self.margin - mask[:, : -1] * diff))
        #loss2 = torch.mean(torch.log(1 + torch.sum(torch.exp(self.margin - mask[:, : -1] * diff), dim = 1)))
        #loss2 = torch.mean(F.softplus(torch.logsumexp(self.margin - mask[:, : -1] * diff, dim = 1)))
        return loss1 + loss2

        #return loss2

if __name__ == "__main__":
    embedding = torch.randn(10, 64)
    num_classes = 10
    labels = torch.randint(0 , num_classes, (10,))
    criterion = ProxyNCALoss(num_classes, 64)
    criterion(embedding , labels)
    print(criterion.get_proxies())
