from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn

__all__ = ["ResNet50" , "ResNet18"]


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, weight_path='weights_models', pretrained=True):
        super(ResNet50, self).__init__()

        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.extractor = nn.Sequential(*list(model.children())[:-2])
        #model.fc = nn.Identity()
        #model.avgpool = GMP_and_GAP()
        num_channels = 2048
        height, width = 7, 7
        self.fc = nn.Linear(num_channels * height * width, 128)
        #self.avgpool = GMP_and_GAP()
        self.avgpool = model.avgpool
        self.model = model
    def forward(self, x):
        # x : 提取到的特征
        x = self.extractor(x)
        feat = self.fc(x.view(x.size(0), -1))
        x = self.avgpool(x)
        return x.view(x.size(0) , -1) , feat

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self):
        super(ResNet18, self).__init__()

        model = resnet18(weights = ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.avgpool = GMP_and_GAP()
        self.model = model
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0) , -1)
        return x


class GMP_and_GAP(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        
        return self.gap(x) + self.gmp(x)