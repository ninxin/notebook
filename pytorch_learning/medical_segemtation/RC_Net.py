import torch
import torch.nn as nn
import torch.nn.functional as F

class con_bn_re(nn.Module):
    def __init__(self, inchan, outchan):
        super(con_bn_re, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(inchan, outchan, kernel_size=3, padding=1),
                                   # nn.BatchNorm2d(outchan),
                                   nn.GroupNorm(outchan, outchan),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

class con_soft_re_pool(nn.Module):
    def __init__(self, inchan, outchan):
        super(con_soft_re_pool, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, padding=1)
        self.re = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        x = self.re(x)
        x = self.pool(x)
        return x

class con_soft(nn.Module):
    def __init__(self, inchan, outchan):
        super(con_soft, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        return x

class re_pool(nn.Module):
    def __init__(self):
        super(re_pool, self).__init__()
        self.block = nn.Sequential(nn.ReLU(),
                                   nn.MaxPool2d(2))
    def forward(self, x):
        return self.block(x)

class skip_con(nn.Module):
    def __init__(self, chann):
        super(skip_con, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(chann, chann, kernel_size=1),
                                   # nn.BatchNorm2d(chann)
                                   nn.GroupNorm(chann, chann)
                                   )

    def forward(self, x):
        return self.block(x)



class RCNet(nn.Module):
    def __init__(self, in_chann, mid_chann, num_class):
        super(RCNet, self).__init__()
        self.to_x1 = con_bn_re(in_chann, mid_chann)
        self.skip1 = con_soft_re_pool(mid_chann, mid_chann)
        self.to_x2 = con_bn_re(mid_chann, mid_chann)
        self.for_add1 = con_soft(mid_chann, mid_chann)
        self.skip2 = re_pool()
        self.to_x3 = con_bn_re(mid_chann, mid_chann)
        self.for_add2 = con_soft(mid_chann, mid_chann)
        self.skip3 = nn.ReLU()
        self.for_add3 = con_bn_re(mid_chann, mid_chann)
        self.for_add4 = con_soft(mid_chann, mid_chann)
        self.skip4 = nn.ReLU()
        self.for_add5 = con_bn_re(mid_chann, mid_chann)
        self.for_add6 = con_soft(mid_chann, mid_chann)
        self.for_add7 = con_bn_re(mid_chann, mid_chann)
        self.last = nn.Sequential(con_bn_re(mid_chann, mid_chann),
                                  nn.Conv2d(mid_chann, num_class, kernel_size=1))

    def forward(self, x):
        x = self.to_x1(x)
        x1 = x
        x = self.skip1(x)
        skip1 = x
        x = self.to_x2(x)
        x2 = x
        x = self.for_add1(x)
        x = x + skip1
        x = self.skip2(x)
        skip2 = x
        x = self.to_x3(x)
        x3 = x
        x = self.for_add2(x)
        x = x + skip2
        x = self.skip3(x)
        skip3 = x
        x = self.for_add3(x)
        x = x + x3
        x = self.for_add4(x)
        x = x + skip3
        x = self.skip3(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear')
        skip4 = x
        x = self.for_add5(x)
        x = x + x2
        x = self.for_add6(x)
        x = x + skip4
        x = self.skip3(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = self.for_add7(x)
        x = x + x1
        out = self.last(x)
        return {'out': out}
