import torch
import torch.nn as nn
import torch.nn.functional as F



class SA(nn.Module):
    def __init__(self, in_channel):
        super(SA, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=[1, in_channel])
        self.avg = nn.AvgPool2d(kernel_size=[1, in_channel])
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        assert x.dim() == 4
        out1 = x
        out2 = x
        out1 = out1.transpose(1, 3)
        out1 = self.max(out1)
        out1 = out1.transpose(1, 3)

        out2 = out2.transpose(1, 3)
        out2 = self.max(out2)
        out2 = out2.transpose(1, 3)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv(out)
        out = self.sig(out)
        logits = x * out

        return logits


class DAC(nn.Module):
    def __init__(self, in_chann):
        super(DAC, self).__init__()
        # dilation = [1, 3, 5]
        self.block1 = nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=1, padding=1)
        self.block2 = nn.Sequential(nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=3, padding=3),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=1))
        self.block3 = nn.Sequential(nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=1, padding=1),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=3, padding=3),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=1))
        self.block4 = nn.Sequential(nn.Conv2d(in_chann, in_chann, kernel_size=3, padding=1),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=3, padding=3),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=3, dilation=5, padding=5),
                                    nn.Conv2d(in_chann, in_chann, kernel_size=1))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out = out1 + out2 + out3 + out4
        return out

class MKP(nn.Module):
    def __init__(self, in_chann):
        super(MKP, self).__init__()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(3)
        self.pool3 = nn.MaxPool2d(5)
        self.pool4 = nn.MaxPool2d(6)
        self.conv1 = nn.Conv2d(in_chann*5, in_chann, kernel_size=1)

    def forward(self, x):
        out = x
        out1 = self.pool1(x)
        out2 = self.pool2(x)
        out3 = self.pool3(x)
        out4 = self.pool4(x)

        out1 = F.interpolate(out1, size=x.shape[2:], mode='bilinear')
        out2 = F.interpolate(out2, size=x.shape[2:], mode='bilinear')
        out3 = F.interpolate(out3, size=x.shape[2:], mode='bilinear')
        out4 = F.interpolate(out4, size=x.shape[2:], mode='bilinear')

        out = torch.cat([out, out1, out2, out3, out4], dim=1)
        out = self.conv1(out)

        return out

class convblock(nn.Module):
    def __init__(self, in_chann, out_chann):
        super(convblock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_chann, out_chann, kernel_size=3, padding=1),
                                   nn.Dropout(0.2),
                                   # nn.BatchNorm2d(out_chann),
                                   nn.GroupNorm(out_chann, out_chann),
                                   nn.ReLU())

    def forward(self, x):
        x = self.block(x)
        return x

class MCUNet(nn.Module):
    def __init__(self, inchan, num_class):
        super(MCUNet, self).__init__()
        self.block1 = convblock(inchan, 64)
        self.block2 = convblock(64, 128)
        self.block3 = convblock(128, 256)

        self.sa = SA(256)
        self.dac = DAC(256)
        self.mkp = MKP(256)

        self.block_1 = convblock(256, 128)
        self.block_2 = convblock(128, 64)
        self.block_3 = convblock(64, 32)

        self.pool = nn.MaxPool2d(2)
        self.lastconv = nn.Conv2d(32, num_class, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.pool(x1)
        x2 = self.block2(x2)
        x3 = self.pool(x2)
        x3 = self.block3(x3)
        out = self.pool(x3)
        out1 = self.sa(out)
        out2 = self.dac(out)
        out2 = self.mkp(out2)
        out = out1 + out2
        out = F.interpolate(out, size=x3.shape[2:], mode='bilinear')
        out = out + x3
        out = self.block_1(out)
        out = F.interpolate(out, size=x2.shape[2:], mode='bilinear')
        out = out + x2
        out = self.block_2(out)
        out = F.interpolate(out, size=x1.shape[2:], mode='bilinear')
        out = out + x1
        out = self.block_3(out)
        out = self.lastconv(out)

        return {'out': out}
