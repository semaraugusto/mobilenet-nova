import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    '''Separable convolution'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.dw_bn(self.dw_conv(x)))
        out = F.relu(self.pw_bn(self.pw_conv(out)))
        return out


class MyMobileNet(nn.Module):
    cfg = [
        (32, 64, 1), 
        (64, 128, 2), 
        (128, 128, 1), 
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 512, 1),
        (512, 1024, 2),
        (1024, 1024, 1),
    ]
    def __init__(self, num_classes=10, alpha: float = 1):
        super(MyMobileNet, self).__init__()
        conv_out = int(32 * alpha)
        self.conv = nn.Conv2d(3, conv_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(conv_out)

        self.features = self.make_feature_extractor(alpha)
        self.linear = nn.Linear(1024, num_classes)

    def make_feature_extractor(self, alpha):
        layer_values = [(int(inp*alpha), int(out*alpha), chan) for inp, out, chan in self.cfg]
        layers = nn.Sequential(*[SeparableConv2d(*tup) for tup in layer_values])
        return layers

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.features(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out
