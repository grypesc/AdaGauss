import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, activation, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.activation(out)


class ResNet(nn.Module):
    activation_function_dict = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(inplace=True),
        "lrelu": nn.LeakyReLU(inplace=True)
    }

    def __init__(self, block, layers, num_features, num_classes=10, activation_function="relu", normalize=False):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        if activation_function not in self.activation_function_dict.keys():
            raise RuntimeError(f"Wrong activation function. Possible choices are {list(self.activation_function_dict.keys())}")
        self.activation = self.activation_function_dict[activation_function]
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        if num_features is None:
            self.bottleneck = None
        else:
            self.bottleneck = nn.Conv2d(64, num_features, 1, stride=1)
        self.normalize = normalize
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='activation')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.activation, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        features = torch.mean(x, dim=(2, 3))
        return features


def resnet32(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model


def resnet20(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model


def resnet14(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    n = 2
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model


def resnet8(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    n = 1
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model
