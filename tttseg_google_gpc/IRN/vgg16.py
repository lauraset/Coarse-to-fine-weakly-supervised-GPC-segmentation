
import torch
import torch.nn as nn

def make_layers(layer, batchnorm, in_channels=3):
    layers = []
    for v in layer:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'D':
            layers += [nn.Dropout(p=0.5)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batchnorm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v, eps=0.001, momentum=0.99)] # reversed
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self, cfg, num_classes=2, batchnorm=True):
        super(VGG, self).__init__()
        self.out_channels = 1024
        self.layer1 = make_layers(cfg[0], batchnorm, in_channels=3)
        self.layer2 = make_layers(cfg[1], batchnorm, in_channels=64)
        self.layer3 = make_layers(cfg[2], batchnorm, in_channels=128)
        self.layer4 = make_layers(cfg[3], batchnorm, in_channels=256)
        self.layer5 = make_layers(cfg[4], batchnorm, in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, num_classes),
            nn.Sigmoid()
        )
        # self._load_pretrained(self.model_dir, self.tag)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg16_bn(num_classes=2, batchnorm=True):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        model_dir (str): Directory holding the pre-trained CNN models, must contain 'models_CNN' at the end
        tag (str): Model name (containing dataset and CNN architecture)
        num_classes (int): Number of output classes
        batchnorm (bool): If True, use batch normalization layers
        pretrained (bool): If True, initializes model with pre-trained weights
    """
    cfg = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 512, 512, 512], [1024, 'D', 1024, 'D']]
    net = VGG(cfg, num_classes=num_classes, batchnorm=batchnorm)

    return net