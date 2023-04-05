import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenetv2

# model = mobilenetv2.mobilenet_v2(pretrained=True)
# model.features[0]   # 3 32
# model.features[1]   # 32 16
# model.features[3]   # 24 32         3 -> 4
# model.features[6]   # 96 160        6 -> 14


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class resblock(nn.Module):
    def __init__(self, inchann, outchann):
        super(resblock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(inchann, outchann//4, kernel_size=1),
                                   nn.BatchNorm2d(outchann//4),
                                   nn.ReLU(),
                                   nn.Conv2d(outchann//4, outchann//4, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(outchann//4),
                                   nn.ReLU(),
                                   nn.Conv2d(outchann//4, outchann, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(outchann))
        self.skip = nn.Sequential(nn.Conv2d(inchann, outchann, kernel_size=1),
                                  nn.BatchNorm2d(outchann))
        self.se = nn.Sequential(nn.ReLU(),
                                SELayer(outchann))


    def forward(self, x):
        out = self.block(x)
        skip = self.skip(x)
        out = out + skip

        return self.se(out)

"""
[32, 64, 128]
[32, 64, 96]
[16, 24, 32]
"""
class NanoNet(nn.Module):
    def __init__(self):
        super(NanoNet, self).__init__()
        # self.encode = mobilenetv2.mobilenet_v2(pretrained=True)
        self.res1 = resblock(32, 48)
        self.res2 = resblock(48+24, 32)
        self.res3 = resblock(32+16, 24)
        self.res4 = resblock(24+3, 16)
        self.skip1 = nn.Sequential(nn.Conv2d(24, 24, kernel_size=1),
                                   nn.BatchNorm2d(24),
                                   nn.ReLU())
        self.skip2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.skip3 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        self.last = nn.Sequential(nn.Conv2d(16, 2, kernel_size=1),
                                  nn.Sigmoid())

    def forward(self, x):
        out = []
        x2 = x

        net = mobilenetv2.mobilenet_v2(pretrained=True)
        net.to('cuda')
        for i in range(7):
            x = net.features[i](x)
            out.append(x)
        out1 = self.res1(out[6])
        out1 = F.interpolate(out1, size=out[3].shape[2:], mode='bilinear')
        out1 = torch.cat([out1, self.skip1(out[3])], dim=1)
        out1 = self.res2(out1)
        out1 = F.interpolate(out1, size=out[1].shape[2:], mode='bilinear')
        out1 = torch.cat([out1, self.skip2(out[1])], dim=1)
        out1 = self.res3(out1)
        out1 = F.interpolate(out1, size=x2.shape[2:], mode='bilinear')
        out1 = torch.cat([out1, self.skip3(x2)], dim=1)
        out1 = self.res4(out1)
        out1 = self.last(out1)
        return {'out': out1}

# model = NanoNet()
# a = torch.rand(1, 3, 256, 256)
# print(model(a).shape)