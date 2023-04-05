import torch
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from alexnet import Alexnet
import os
import json
import time

import torch.nn as nn
# import pickle
# from torch.utils.data import Dataset
# from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    'train': transforms.Compose([transforms.CenterCrop((5000, 2600)),
                                 transforms.Grayscale(),
                                transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.5), (0.5))
                                 ]),
    'val': transforms.Compose([transforms.CenterCrop((5000, 2600)),
                                transforms.Grayscale(),
                              transforms.ToTensor(),
                              # transforms.Normalize((0.5), (0.5))
                               ])
}

# class beidu_dataset(Dataset):
#     def __init__(self, path, mode, is_val=False, split=None):
#
#         self.mode = mode
#         self.is_val = is_val
#         self.data_path = os.path.join(path, f"{mode}_pro")
#         self.data_file = os.listdir(self.data_path)
#         self.img_file = self._select_img(self.data_file)
#         if split is not None and mode == "training":
#             assert split > 0 and split < 1
#             if not is_val:
#                 self.img_file = self.img_file[:int(split*len(self.img_file))]
#             else:
#                 self.img_file = self.img_file[int(split*len(self.img_file)):]
#         self.transforms = Compose([
#             RandomHorizontalFlip(p=0.5),
#             RandomVerticalFlip(p=0.5),
#         ])
#
#     def __getitem__(self, idx):
#         img_file = self.img_file[idx]
#         with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
#             img = torch.from_numpy(pickle.load(file)).float()
#         gt_file = "gt" + img_file[3:]
#         with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
#             gt = torch.from_numpy(pickle.load(file)).float()
#
#         if self.mode == "training" and not self.is_val:
#             seed = torch.seed()
#             torch.manual_seed(seed)
#             img = self.transforms(img)
#             torch.manual_seed(seed)
#             gt = self.transforms(gt)
#
#         return img, gt
#
#     def _select_img(self, file_list):
#         img_list = []
#         for file in file_list:
#             if file[:3] == "img":
#                 img_list.append(file)
#
#         return img_list
#
#     def __len__(self):
#         return len(self.img_file)


# data_root = os.path.abspath('D:/pythonlearning/torch')
data_root = "/mnt/d/pythonlearning/torch/"
image_path = data_root + '/dataset/'
train_dataset = datasets.ImageFolder(root=image_path + '/train', transform=data_transform['train'])
# train_dataset = beidu_dataset(image_path, 'train')
train_num = len(train_dataset)
# print(train_num)

# data_list = train_dataset.class_to_idx
# cla_dict = dict((val, key) for key, val in data_list.items())
# cla_dict = {'':}

# json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

validate_dataset = datasets.ImageFolder(root='/mnt/d/pythonlearning/torch/dataset/val', transform=data_transform['val'])
# validate_dataset = beidu_dataset(image_path, 'val')
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = Alexnet(num_classes=2, init_weights=False)

net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './Alexnet2.pth'
best_acc = 0.0
for epoch in range(20):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = '*' * int(rate*50)
        b = '.' * int((1-rate)*50)
        print('\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}'.format(int(rate*100), a, b, loss), end='')
    # print()
    print(time.perf_counter()-t1)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_image, test_label = data_test
            outputs = net(test_image.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_label.to(device)).sum().item()
        accurate_test = acc / (val_num+0.0001)
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / step, acc / val_num))

print('finished training')



