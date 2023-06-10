from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

__all__ = ["ResNet50"]


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, weight_path='weights_models', pretrained=True):
        super(ResNet50, self).__init__()

        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)        

        del self.model.fc
        self.model.avgpool = GMP_and_GAP()

    def forward(self, x):
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0) , -1)
        return x


class GMP_and_GAP(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        
        return self.gap(x) + self.gmp(x)