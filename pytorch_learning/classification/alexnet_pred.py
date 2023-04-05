import torch
from alexnet import Alexnet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.CenterCrop((5000, 2600)),
    transforms.Grayscale(),
     transforms.ToTensor(),
     # transforms.Normalize((0.5), (0.5))
     ]
)

img = Image.open('/mnt/d/pythonlearning/torch/dataset/0201/wubeidu/2.tif')
# plt.imshow(img)

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)       # 增加一个维度放batch

try:
    json_file = open('/mnt/d/pythonlearning/torch/class_indices.json', 'r')        # 加载
    class_indict = json.load(json_file)                 # 解码成字典
except Exception as e:
    print(e)
    exit(-1)

model = Alexnet(num_classes=2)

model_weigght_path = '/mnt/d/pythonlearning/torch/Alexnet2.pth'
model.load_state_dict(torch.load(model_weigght_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))          # 将batch压缩掉
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
# plt.show()