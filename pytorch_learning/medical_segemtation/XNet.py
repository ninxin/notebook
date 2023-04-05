import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import EfficientNet
"""
shift+enter     跳到下一行
ctrl+右箭头      右跳一个单词  
alt+右箭头       右跳一个文件
ctrl+d          复制这一行到下一行
ctrl+s          保存
ctrl+w          关闭当前页面
ctrl+n          新建一个文件
ctrl+f          搜索当前页面
ctrl+shift+f    搜索所有页面
ctrl+r          替换
ctrl+shift+r    全局替换
"""

class DropBlock2d(nn.Module):
    """
    Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.
    Args:
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``
    """

    def __init__(self, p: float, block_size: int, inplace: bool = False) -> None:
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s


class conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv, self).__init__()
        self.con = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
                                 nn.GroupNorm(out_channel, out_channel),  # 将通道分成几组， 输入的通道数
                                 # nn.ELU(),
                                 nn.ReLU()
                                 )

    def forward(self, x):
        return self.con(x)


""" Squeeze and Excitation block """


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Hardswish()
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvTranspose(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride,
                 padding,
                 **kwargs):
        super(ConvTranspose, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.transpose = nn.ConvTranspose2d(in_channels=self.input_channels,
                                            out_channels=self.output_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=kernel_size // 2,
                                            **kwargs)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor
                ) -> torch.Tensor:
        out = self.transpose(x1)
        diffY = x2.size()[2] - out.size()[2]
        diffX = x2.size()[3] - out.size()[3]

        out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        out = torch.cat([x2, out], dim=1)
        return out


"""
2 
先上采样在下采样, 
一起使用maxpool,avgpool, 
替换卷积块, 替换跳跃链接, 
对解码器再利用 
SE块

"""


class ConvBnRelu(nn.Module):
    def __init__(self, inchan, outchan, kernel, pad=0, group=1):
        super(ConvBnRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchan, outchan, kernel_size=kernel, padding=pad, groups=group, bias=False),
            # nn.BatchNorm2d(outchan),
            nn.GroupNorm(outchan, outchan),
            nn.ReLU(inplace=True))


    def forward(self, x):
        return self.block(x)


class my_block(nn.Module):
    def __init__(self, inchan, outchan, use_group=True):
        super(my_block, self).__init__()

        if use_group:
            group = 4
        else:
            group = 1

        self.conv1 = ConvBnRelu(inchan, outchan, kernel=3, pad=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(outchan, outchan, kernel_size=(1, 5), padding=2, bias=False, groups=group),
            nn.Conv2d(outchan, outchan, kernel_size=(5, 1), bias=False, groups=group),
            # nn.BatchNorm2d(outchan),
            nn.GroupNorm(outchan, outchan),
            nn.ReLU(inplace=True))

        self.block2 = nn.Sequential(
            nn.Conv2d(outchan, outchan, kernel_size=(1, 7), padding=3, bias=False, groups=group),
            nn.Conv2d(outchan, outchan, kernel_size=(7, 1), bias=False, groups=group),
            # nn.BatchNorm2d(outchan),
            nn.GroupNorm(outchan, outchan),
            nn.ReLU(inplace=True))

        self.block3 = nn.Sequential(nn.Conv2d(outchan, outchan, kernel_size=3, padding=1, bias=False, groups=group),
                                    # nn.BatchNorm2d(outchan),
                                    nn.GroupNorm(outchan, outchan),
                                    nn.ReLU(inplace=True))

        self.block4 = nn.Sequential(nn.Conv2d(outchan, outchan, kernel_size=1, bias=False, groups=group),
                                    # nn.BatchNorm2d(outchan),
                                    nn.GroupNorm(outchan, outchan),
                                    nn.ReLU(inplace=True))

        self.conv2 = ConvBnRelu(outchan * 4, outchan, kernel=1)

        if inchan == outchan:
            self.conv3 = nn.Identity()
        else:
            self.conv3 = ConvBnRelu(inchan, outchan, kernel=1)

        self.finalconv = ConvBnRelu(outchan, outchan, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        output = torch.cat([self.block1(x1), self.block2(x1), self.block3(x1), self.block4(x1)], dim=1)
        output = self.conv2(output)
        output = output + self.conv3(x)
        return self.finalconv(output)

class MyNet(nn.Module):
    def __init__(self, inchan, outclass):
        super(MyNet, self).__init__()
        # 3-64, 64-128, 128-256, 256-256
        self.down1 = my_block(inchan, 64, use_group=False)
        self.down2 = my_block(64, 128, use_group=True)
        self.down3 = my_block(128, 256, use_group=True)
        self.down4 = my_block(256, 256, use_group=True)
        self.pool = nn.AvgPool2d(2)

        self.up1 = ConvBnRelu(256, 256, kernel=1)
        self.up2 = ConvBnRelu(256, 256, kernel=1)
        self.up3 = ConvBnRelu(256, 128, kernel=1)
        self.up4 = ConvBnRelu(128, 64, kernel=1)

        self.final = ConvBnRelu(64, outclass, kernel=1)

    def forward(self, x):
        x1 = self.down1(x)      # 1, 64
        x2 = self.pool(x1)      # 1/2, 64
        x2 = self.down2(x2)     # 1/2, 128
        x3 = self.pool(x2)      # 1/4, 128
        x3 = self.down3(x3)     # 1/4, 256
        x4 = self.pool(x3)      # 1/8, 256
        x4 = self.down4(x4)     # 1/8, 256

        out = self.up1(x4) + x4     # 1/8, 256
        out = F.interpolate(out, size=x3.shape[2:], mode='bilinear')        # 1/4, 256
        out = self.up2(out) + x3        # 1/4, 256
        out = F.interpolate(out, size=x2.shape[2:], mode='bilinear')        # 1/2, 256
        out = self.up3(out) + x2        # 1/2, 128
        out = F.interpolate(out, size=x1.shape[2:], mode='bilinear')        # 1, 64
        out = self.up4(out)             # 1, 64
        out = self.final(out)
        return {'out': out}
