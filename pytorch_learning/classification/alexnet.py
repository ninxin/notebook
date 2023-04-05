import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(8),                                    # 625 325
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(16, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=4),              # 312 162
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=4),              # 156 81
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(64, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(128, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(128, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),              # 78 40
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(20736, 2048),       # 6*6
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# net = Alexnet(2)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# a = torch.randn(1, 1, 6000, 2600).to(device)
# net.to(device)
# print(net(a).shape)