import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from resneXt import resneXt50_32x4d

# import torchvision.models.resnet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

data_root = os.path.abspath('D:/pythonlearning/torch')
image_path = data_root + '/flower_data/'
train_dataset = datasets.ImageFolder(root=image_path + '/train', transform=data_transform['train'])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validate_dataset = datasets.ImageFolder(root='D:/pythonlearning/torch/flower_data/val', transform=data_transform['val'])
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

net = resneXt50_32x4d()

# load pretain weights
model_weight_path = './resnext50_32x4d.pth'
assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

# 冻结除最后一个连接层外的参数，只训练最后一层
for param in net.parameters():
    param.requires_grad = False

inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)            # 将1000个类变成5个类，全连接层输出5
net.to(device)

loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0001)

save_path = './resneXt50.pth'
best_acc = 0.0
for epoch in range(3):
    net.train()
    running_loss = 0.0
    #t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = '*' * int(rate*50)
        b = '.' * int((1-rate)*50)
        print('\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}'.format(int(rate*100), a, b, loss), end='')
    print()
    #print(time.perf_counter()-t1)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_image, test_label = data_test
            outputs = net(test_image.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_label.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / step, acc / val_num))

print('finished training')