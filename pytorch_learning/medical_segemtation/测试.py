from UCTransNet import UCTransNet
import torch

x = torch.zeros(3, 500, 500)
model = UCTransNet()
model(x)