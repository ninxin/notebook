import torch
import torchvision.transforms as transforms
from PIL import Image
from lenet import Lenet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Lenet()
net.load_state_dict(torch.load('lenet.pth'))

im = Image.open('图片地址')
im = transform(im)
im = torch.unsqueeze(im, dim=0)

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])