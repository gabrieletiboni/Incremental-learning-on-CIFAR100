#remove ReLU in the last layer, and use cosine layer to replace nn.Linear

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, norm_during_training=False):

        self.inplanes = 16
        super(ResNet, self).__init__()

        # First conv layer
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Add 3 layers composed of layers[i]-BasicBlock each (each BasicBlock has 2 layers)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2) # a downsample layer (conv1x1) is added in the first BasicBlock to adjust input to the output lower dimension caused by stride=2
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last=True) # a downsample layer (conv1x1) is added in the first BasicBlock to adjust input to the output lower dimension caused by stride=2

        # AVG pool on each feature map so output is size 1x1 with depth 64
        self.avgpool = nn.AvgPool2d(8, stride=1)

        # Cosine Norm
        self.linear = CosineLinear(64 * block.expansion, num_classes)
        #nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='sigmoid')

        self.out_dim = 64 * block.expansion
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        if not last:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            # Skip last ReLu on the very last block
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))

        return nn.Sequential(*layers)

    def feature_map(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x

    def L2_norm(self, features): 
        # L2-norm on rows
        features_norm = torch.zeros((features.size(0),features.size(1)), dtype=torch.float32).to('cuda')

        for i,feature in enumerate(features):
            square = torch.square(feature)
            somma = torch.sum(square)
            sqrt = torch.sqrt(somma).item()
            features_norm[i] += feature/sqrt

        return features_norm

class CosineLinear(nn.Module):	
    def __init__(self, in_features, out_features, sigma=True):	
        super(CosineLinear, self).__init__()	
        self.in_features = in_features	
        self.out_features = out_features	
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))	
        if sigma:	
            self.sigma = nn.Parameter(torch.Tensor(1))	
        else:	
            self.register_parameter('sigma', None)	
        print(self.sigma)
        self.reset_parameters()	

    def reset_parameters(self):	
        stdv = 1. / math.sqrt(self.weight.size(1))	
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:	
            self.sigma.data.fill_(1) #for initializaiton of sigma	

    def forward(self, input):	
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))	
        if self.sigma is not None:	
            out = self.sigma * out	
        return out

# def resnet20(pretrained=False, **kwargs):
#     n = 3
#     model = ResNet(BasicBlock, [n, n, n], **kwargs)
#     return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model