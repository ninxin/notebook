import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from mobilenet import MobileNetV2


# https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
# import torchvision.models.mobilenetv2


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

net = MobileNetV2(num_classes=5)
# load pretain weights
model_weight_path = './mobilenet_v2.pth'
pre_weights = torch.load(model_weight_path)     # 得到字典类型
# 在imagenet数据集上训练的，最后一层的全连接层的节点是1000，而我们的是5，不能用
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}    # 遍历权重字典，最后一层全连接层叫classifier，如果不叫这个就保存
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)     # 将权重字典载入

# 冻结特征提取的所有权重
for param in net.features.parameters():
    param.requires_grad = False

net.to(device)

loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0001)

save_path = './MobileNetV2.pth'
best_acc = 0.0
for epoch in range(5):
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