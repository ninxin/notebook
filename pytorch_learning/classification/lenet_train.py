import torch
import torchvision
import torch.nn as nn
from lenet import Lenet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# totensor将H*W*C变成C*H*W，并将数值0-255变成0.0-1.0
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR_data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./CIFAR_data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5         # 将标准化后的数据转换回原来的
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))      # 转化成原来的格式，将C*H*W变成H*W*C
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))

net = Lenet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)       # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item()/test_label.size(0)        # item得到相等的个数

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('finish training')

save_path = './lenet.pth'
torch.save(net.state_dict(), save_path)
