"""RevNet implementation used for The Reversible Residual Network
Implemented in PyTorch instead of TensorFlow.

@inproceedings{gomez17revnet,
  author    = {Aidan N. Gomez and Mengye Ren and Raquel Urtasun and Roger B. Grosse},
  title     = {The Reversible Residual Network: Backpropagation without Storing Activations}
  booktitle = {NIPS},
  year      = {2017},
}

Github: https://github.com/renmengye/revnet-public

Author: Sil van de Leemput

"""
import torch
import torch.nn as nn
import math
from .revop import ReversibleBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def batch_norm(input):
    """match Tensorflow batch norm settings"""
    return nn.BatchNorm2d(input, momentum=0.99, eps=0.001)


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(RevBasicBlock, self).__init__()
        if downsample is None and stride == 1:
            Gm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            Fm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            self.revblock = ReversibleBlock(Gm, Fm)
        else:
            self.basicblock_sub = BasicBlockSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.shortcut = nn.AvgPool2d(2)
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            out = self.basicblock_sub(x)
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)], 1)
        else:
            out = self.revblock(x)
        return out


class BasicBlockSub(nn.Module):
    def __init__(self, inplanes, planes, stride=1, noactivation=False):
        super(BasicBlockSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = batch_norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class RevNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, channels_per_layer=None, strides=None, init_max_pool=False, init_kernel_size=7, batch_norm_fix=True, implementation=0):
        if channels_per_layer is None:
            channels_per_layer = [2 ** (i + 6) for i in range(len(layers))]
            channels_per_layer = [channels_per_layer[0]] + channels_per_layer
        if strides is None:
            strides = [2] * len(channels_per_layer)
        self.batch_norm_fix = batch_norm_fix
        self.channels_per_layer = channels_per_layer
        self.strides = strides
        self.init_max_pool = init_max_pool
        self.implementation = implementation
        assert(len(self.channels_per_layer) == len(layers) + 1)
        self.inplanes = channels_per_layer[0]  # 64 by default
        super(RevNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=init_kernel_size, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = batch_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.init_max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels_per_layer[1], layers[0], stride=strides[1], noactivation=True)
        self.layer2 = self._make_layer(block, channels_per_layer[2], layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, channels_per_layer[3], layers[2], stride=strides[3])
        self.has_4_layers = len(layers) >= 4
        if self.has_4_layers:
            self.layer4 = self._make_layer(block, channels_per_layer[4], layers[3], stride=strides[4])
        self.bn_final = batch_norm(self.inplanes)  # channels_per_layer[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels_per_layer[-1] * block.expansion, num_classes)

        self.configure()
        self.init_weights()

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    def configure(self):
        """Initialization specific configuration settings"""
        for m in self.modules():
            if isinstance(m, ReversibleBlock):
                m.implementation = self.implementation
            elif isinstance(m, nn.BatchNorm2d):
                if self.batch_norm_fix:
                    m.momentum = 0.99
                    m.eps = 0.001
                else:
                    m.momentum = 0.1
                    m.eps = 1e-05

    def _make_layer(self, block, planes, blocks, stride=1, noactivation=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, noactivation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.init_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.has_4_layers:
            x = self.layer4(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def RevNet18(n_classes):
    return RevNet(block=RevBasicBlock, layers=[2, 2, 2, 2], init_kernel_size=3, strides=[1, 1, 2, 2, 2], channels_per_layer=[64, 64, 128, 256, 512], num_classes=n_classes)
