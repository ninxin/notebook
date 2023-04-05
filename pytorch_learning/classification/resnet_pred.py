import torch
from resnet import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

img = Image.open('图片地址')
plt.imshow(img)

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)       # 增加一个维度放batch

try:
    json_file = open('/class_indices.json', 'r')        # 加载
    class_indict = json.load(json_file)                 # 解码成字典
except Exception as e:
    print(e)
    exit(-1)

model = resnet34(num_classes=5)

model_weigght_path = '/resnet34.pth'
model.load_state_dict(torch.load(model_weigght_path))
model.eval()
with torch.no_grad:
    output = torch.squeeze(model(img))          # 将batch压缩掉
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()