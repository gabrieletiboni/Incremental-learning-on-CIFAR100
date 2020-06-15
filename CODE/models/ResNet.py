# resnet_cifar.py fornita nel forum da Cermelli
# modifica

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, norm_during_training=False):

        self.inplanes = 16
        super(ResNet, self).__init__()

        # First conv layer
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Add 3 layers composed of layers[i]-BasicBlock each (each BasicBlock has 2 layers)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2) # a downsample layer (conv1x1) is added in the first BasicBlock to adjust input to the output lower dimension caused by stride=2
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2) # a downsample layer (conv1x1) is added in the first BasicBlock to adjust input to the output lower dimension caused by stride=2

        # AVG pool on each feature map so output is size 1x1 with depth 64
        self.avgpool = nn.AvgPool2d(8, stride=1)

        # Last and only FC layer
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
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
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

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

# def resnet20(pretrained=False, **kwargs):
#     n = 3
#     model = ResNet(BasicBlock, [n, n, n], **kwargs)
#     return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model