# Refer to https://blog.csdn.net/winycg/article/details/86709991
# as basic architecture
# Refer to https://blog.csdn.net/weixin_39675215/article/details/111640830
# for Xavier initialization
# Note that this code is not used in the training of best model, use effnet instead

import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差块，2个3*3卷积，用于ResNet18和34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_dim, out_dim, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.shortcut = nn.Sequential()
        # 若in和out纬度不同，需要添加卷积&BN来变换为同一维度
        if stride != 1 or in_dim != self.expansion * out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, self.expansion * out_dim,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_dim)
            )

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.constant_(self.bn1.bias, 0)
        torch.nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 残差块，1*1+3*3+1*1卷积，用于ResNet50，101和152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(out_dim, self.expansion * out_dim,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != self.expansion * out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, self.expansion * out_dim,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_dim)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_dim = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048 * block.expansion, num_classes)

    def _make_layer(self, block, out_dim, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, out_dim, stride))
            self.in_dim = out_dim * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Resnet152
def Resnet():
    return ResNet(BasicBlock, [2,2,2,2])

def Net():
    return Resnet()