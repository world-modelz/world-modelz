
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Residual(nn.Module):
    def __init__(self, in_planes, hidden_planes, stride=1, normalize=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            conv3x3(in_planes=in_planes, out_planes=hidden_planes, stride=stride),
            normalize(hidden_planes),
            nonlinearity(inplace=True),
            conv1x1(in_planes=hidden_planes, out_planes=in_planes),
            normalize(in_planes)
        )

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=stride, stride=stride, bias=False),
                normalize(in_planes),
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self._block(x) + residual
        return F.leaky_relu(x)


class ResidualStack(nn.Module):
    def __init__(self, in_planes, num_layers, hidden_planes):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(Residual(in_planes, hidden_planes, 1))
            layers.append(Residual(in_planes, hidden_planes, 2))
        self._stack = nn.Sequential(*layers)
        
    def forward(self, x):
        return self._stack(x)


class SimpleResidualEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, num_layers, hidden_planes):
        super(SimpleResidualEncoder, self).__init__()
        self._conv_1 = conv3x3(in_planes, out_planes)
        self._residual_stack = ResidualStack(in_planes=out_planes,
                                             num_layers=num_layers,
                                             hidden_planes=hidden_planes)
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            """
            if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear) 
                or isinstance(m, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
                #nn.init.orthogonal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif """
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self._conv_1(x)
        x = F.leaky_relu(x, inplace=True)
        return self._residual_stack(x)


class UpscaleResidual(nn.Module):
    def __init__(self, in_planes, out_planes, upsample):
        super(UpscaleResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.upsample = upsample
        self.learn_conv_residual = in_planes != out_planes or upsample
        if self.learn_conv_residual:
            self.conv_residual = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            """
            if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear) 
                or isinstance(m, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
                #nn.init.orthogonal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            """
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.bn1(x)
        h = self.act1(h)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.act2(h)
        h = self.conv2(h)
        if self.learn_conv_residual:
            x = self.conv_residual(x)
        return h + x


class SimpleResidualDecoder(nn.Module):
    def __init__(self, cfg, in_channels, out_channels=3):
        super(SimpleResidualDecoder, self).__init__()

        upsample = functools.partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=False)

        layers = [conv3x3(in_channels, in_channels)]
        for hidden_channels in cfg:
            layers += [UpscaleResidual(in_channels, hidden_channels, upsample)]
            in_channels = hidden_channels

        # add output mapping
        layers += [conv3x3(in_channels, out_channels)]

        self.decoder_stack = nn.Sequential(*layers)
 
    def forward(self, x):
        y = self.decoder_stack(x)
        return y
