from .backbone import *
from .embedding import Embedding
from .mlp import MLP
import torch.nn as nn

__all__ = ['MultiTaskModel']
class MultiTaskModel(nn.Module):
    def __init__(self ,  f_dim, g_dim, g_hidden_size ,Backbone = "resnet50"):
        super(MultiTaskModel, self).__init__()
        BackboneOptions = {
            'resnet18' : ResNet18(),
            'resnet50': ResNet50(),
        }
        self.backbone = None
        try:
            self.backbone = BackboneOptions[Backbone]
        except KeyError:
            print(f"The Backbone must be one of {list(BackboneOptions.keys())}")
        #TODO : Other baseline
        output_size = self.backbone.output_size
        self.f_head = Embedding(output_size , f_dim)
        self.g_head = MLP(feature_size=output_size, embedding_size=g_dim , hidden_size=g_hidden_size)


    def forward(self , x, tag = 0):
        x, feat = self.backbone(x)
        if tag == 0:
            return self.f_head(x)
        else:
            return self.f_head(x) , feat
        #return feature , output

if __name__ == "__main__":
    model = Model(128 , 128, 512)
    print(
        model.parameters() ,
        model.backbone.parameters()
    )