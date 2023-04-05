import torch
import torch.nn as nn
import torch.nn.functional as F

class con_bn_re(nn.Module):
    def __init__(self, inchann):
        super(con_bn_re, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inchann, 8, kernel_size=3, padding=1),
                                  # nn.BatchNorm2d(8),
                                  nn.GroupNorm(8, 8),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))

    def forward(self, x):
        return self.conv(x)

class Exconv(nn.Module):
    def __init__(self, inchann, rate=1, skip=False, outchann=None):
        super(Exconv, self).__init__()

        if outchann == None:
            outchann = inchann
            skip = True
        self.skip = skip
        self.block = nn.Sequential(nn.Conv2d(inchann, outchann, kernel_size=1),
                                   # nn.BatchNorm2d(outchann),
                                   nn.GroupNorm(outchann, outchann),
                                   nn.ReLU(),
                                   nn.Conv2d(outchann, outchann, kernel_size=3, dilation=rate, padding=rate, stride=1),
                                   # nn.BatchNorm2d(outchann),
                                   nn.GroupNorm(outchann, outchann),
                                   nn.ReLU(),
                                   nn.Conv2d(outchann, outchann, kernel_size=1),
                                   # nn.BatchNorm2d(outchann)
                                   nn.GroupNorm(outchann, outchann),
                                   )
    def forward(self, x):
        out = x
        if self.skip:
            return self.block(x) + out
        else:
            return self.block(x)

class quickattention(nn.Module):
    def __init__(self, chann):
        super(quickattention, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(chann, chann, kernel_size=1),
                                   nn.Sigmoid())
    def forward(self, x):
        out = x
        return out + self.block(x)

class ASPP(nn.Module):
    def __init__(self, chann):
        super(ASPP, self).__init__()
        self.con1 = nn.Conv2d(chann, chann, kernel_size=1)
        self.conv3_6 = nn.Conv2d(chann, chann, kernel_size=3, dilation=6, padding=6)
        self.conv3_12 = nn.Conv2d(chann, chann, kernel_size=3, dilation=12, padding=12)
        self.conv3_18 = nn.Conv2d(chann, chann, kernel_size=3, dilation=18, padding=18)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(chann, chann, kernel_size=1))
        self.conv = nn.Conv2d(chann*5, chann, kernel_size=1)
    def forward(self, x):
        out1 = self.con1(x)
        out2 = self.conv3_6(x)
        out3 = self.conv3_12(x)
        out4 = self.conv3_18(x)
        out5 = self.pool(x)
        out5 = F.interpolate(out5, size=x.shape[2:], mode='bilinear')
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.conv(out)

class HistoSeg(nn.Module):
    def __init__(self):
        super(HistoSeg, self).__init__()
        self.con = con_bn_re(inchann=3)
        self.excon1_6 = nn.Sequential(Exconv(inchann=8, outchann=16),
                                      nn.MaxPool2d(2),
                                      Exconv(inchann=16, outchann=24),
                                      nn.MaxPool2d(2),
                                      Exconv(inchann=24),
                                      Exconv(inchann=24, outchann=32),
                                      nn.MaxPool2d(2),
                                      Exconv(inchann=32),
                                      Exconv(inchann=32))
        # self.excon1 = Exconv(inchann=8, outchann=16)
        # self.excon2 = Exconv(inchann=16, outchann=24)
        # self.excon3 = Exconv(inchann=24)
        # self.excon4 = Exconv(inchann=24, outchann=32)
        # self.excon5 = Exconv(inchann=32)
        # self.excon6 = Exconv(inchann=32)
        self.qa = quickattention(32)
        self.excon7_16 = nn.Sequential(Exconv(inchann=32, outchann=64),
                                       Exconv(inchann=64),
                                       Exconv(inchann=64),
                                       Exconv(inchann=64, outchann=96),
                                       Exconv(inchann=96),
                                       Exconv(inchann=96),
                                       Exconv(inchann=96, outchann=160),
                                       Exconv(inchann=160),
                                       Exconv(inchann=160),
                                       Exconv(inchann=160, outchann=256))
        # self.excon7 = Exconv(inchann=32, outchann=64)
        # self.excon8 = Exconv(inchann=64)
        # self.excon9 = Exconv(inchann=64)
        # self.excon10 = Exconv(inchann=64, outchann=96)
        # self.excon11 = Exconv(inchann=96)
        # self.excon12 = Exconv(inchann=96)
        # self.excon13 = Exconv(inchann=96, outchann=160)
        # self.excon14 = Exconv(inchann=160)
        # self.excon15 = Exconv(inchann=160)
        # self.excon16 = Exconv(inchann=160, outchann=256)
        self.aspp = ASPP(chann=256)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.con1x1 = nn.Conv2d(256*2, 32, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.last = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        out = self.con(x)
        out = self.excon1_6(out)
        out = self.qa(out)
        skip1 = out
        out = self.excon7_16(out)
        out = self.aspp(out)
        skip2 = out
        out = self.GAP(out)
        out = F.interpolate(out, size=skip2.shape[2:], mode='bilinear')
        out = torch.cat([out, skip2], dim=1)
        out = self.con1x1(out)
        out = out + skip1
        out = self.sigmoid(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear')
        return {'out': self.last(out)}
