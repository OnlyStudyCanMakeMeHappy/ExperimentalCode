from pytorch_metric_learning.losses import TripletMarginLoss,MarginLoss, MultiSimilarityLoss
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch

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
    loss_funcA = TripletMarginLoss(distance=dist_pair, margin = args.vartheta)
    loss_funcB = TripletMarginLoss(margin = args.delta)
    return loss_funcA, loss_funcB