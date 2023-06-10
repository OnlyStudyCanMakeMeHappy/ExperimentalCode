import torch
from pytorch_metric_learning import losses, distances, reducers, miners
import torch.nn.functional as F
import torch.nn as nn


# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


class RankingLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=0.5, s=12, Lambda=1.0):
        super(RankingLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.Lambda = Lambda

    """ 
     def forward(self, inputs , aug_inputs):


         assert inputs.size(0) == len(aug_inputs), f"inputs."
         inputs = x[0]
         aug_inputs = x[1:-1]
         M = inputs.size(0)
         sort_loss = 0.0
         pos_loss = 0.0
         for embedding in inputs:
             sort_LSE_loss = 0.0
             pos_LSE_loss = 0.0
             for aug_embedding_list in aug_inputs:
                 pre = torch.zeros(1)
                 sort_LSE_part_loss = 0.0
                 pos_LSE_part_loss = 0.0
                 for j, aug_embedding in enumerate(aug_embedding_list):
                     if j == 0:
                         pre = torch.dot(embedding, aug_embedding)
                     else:
                         cur = torch.dot(embedding, aug_embedding)
                         sort_LSE_part_loss += torch.exp(self.s * (cur - pre + self.alpha))
                         pre = cur
                     pos_LSE_part_loss += torch.exp(self.s * (self.beta - pre))
                 sort_LSE_loss += 1. / self.s * torch.log(1.0 + sort_LSE_part_loss)
                 pos_LSE_loss += 1. / self.s * torch.log(1.0 + pos_LSE_part_loss)
             sort_loss += sort_LSE_loss
             pos_loss += pos_LSE_loss

         sort_loss /= M
         pos_loss /= M

         return sort_loss + self.Lambda * pos_loss
 """
    def forward(self, input : torch.Tensor , aug_inputs : torch.Tensor):
        """
        :param input: (batch_size , dim)
        :param aug_inputs: (N - 1, batch_size, dim)
        :return: torch.Tensor
        """
        #print(input, aug_inputs)
        m = input.size(0)
        # 广播，元素按位相乘
        product = torch.mul(input, aug_inputs)
        # 计算每个二维张量的和堆叠到一列 -> (N - 1, batch_size) -> 每一列是样本x_i分别与x_i1...x_iN的相似度得分矩阵
        Score = torch.sum(product , dim = 2).cpu()
        #
        new_row = torch.zeros((1, m))
        # 计算ranking list loss
        d = torch.cat([new_row, self.s * (Score.diff(dim = 0) + self.alpha)], dim = 0)
        loss1 = torch.mean(torch.logsumexp(d, dim = 0) / self.s)

        # 计算positive constraint
        exponent = torch.cat([new_row, self.s * (self.beta - Score)], dim = 0)
        loss2 = torch.mean(torch.logsumexp(exponent, dim = 0) / self.s)
        # loss1 + loss2 * λ
        return loss1 + loss2 * self.Lambda




def TripleLoss(margin=0.05, miner=None):
    loss_func = losses.TripletMarginLoss(margin=margin, triplets_per_anchor="all")
    return loss_func


def MultiSmimilarityLoss(alpha=2, beta=50, base=0.5, miner=None):
    loss_func = losses.MultiSimilarityLoss(alpha, beta, base)
    return loss_func


def MarginLoss():
    loss_func = losses.MarginLoss()
    return loss_func

if __name__ == "__main__":
    N = 4
    embedding_size = 64
    batch_size = 20
    criterion = RankingLoss()
    inputs = torch.randn(batch_size, embedding_size)
    inputs = F.normalize(inputs, dim=1, p=2)
    aug_inputs = F.normalize(torch.randn(N, batch_size, embedding_size), p=2, dim = 2)
    loss = criterion(inputs, aug_inputs)
    print(loss)
