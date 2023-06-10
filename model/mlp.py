import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, feature_size, embedding_size, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(feature_size, hidden_size)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_size, embedding_size)
        self.hidden.apply(init_weights)
        self.output.apply(init_weights)

    def forward(self, x):
        x = self.act(self.hidden(x))
        return F.normalize(self.output(x) , p = 2, dim = 1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
