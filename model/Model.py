from .backbone import *
import embedding
import mlp
import torch.nn as nn

class Model(nn.Module):
    def __init__(self , Backbone = "resnet50", q_dim, g_dim, g_hidden_size):
        super(Model, self).__init__()
        if Backbone == "resnet50":
            self.backbone = ResNet50
        elif Backbone == "bninception":
            pass
        output_size = self.Backbone.output_size
        self.f = embedding(output_size , q_dim)
        self.g = mlp(output_size, output_size , g_hidden_size , g_dim)


    def forward(self , x , task_id):
        feature = self.backbone(x)
        if task_id == 0:
            output = self.f(feature)
        elif task_id == 1:
            output = self.g(feature)
        else:
            raise Exception("Only two task,The task_id must be 0 or 1!")
        return output