import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
# from ruamel.yaml import safe_load
from ruamel_yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor, CenterCrop


def data_process(data_path, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    # save_path = os.path.join('/mnt/d/pythonlearning/paper', f"{mode}_pro")
    a_path = os.path.join(data_path, mode, "youbeidu")
    b_path = os.path.join(data_path, mode, "wubeidu")

    a_list = list(sorted(os.listdir(a_path)))
    b_list = list(sorted(os.listdir(b_path)))

    you_list = []
    wu_list = []
    for i, file in enumerate(a_list):

        img = Image.open(os.path.join(a_path, file))
        tmp = CenterCrop((6000, 2800))
        img = tmp(img)
        img = Grayscale(1)(img)
        you_list.append(ToTensor()(img))

    for i, file in enumerate(b_list):
        img = Image.open(os.path.join(b_path, file))
        tmp = CenterCrop((6000, 2800))
        img = tmp(img)
        img = Grayscale(1)(img)
        wu_list.append(ToTensor()(img))

    if mode == "train":
        save_patch(you_list, save_path, "you")
        save_patch(wu_list, save_path, "wu")
    elif mode == "val":
        save_each_image(you_list, save_path, "you")
        save_each_image(wu_list, save_path, "wu")


def save_patch(imgs_list, path, type):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save  {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save  {type} : {type}_{i}.pkl')




if __name__ == '__main__':

    data_process("/mnt/d/pythonlearning/torch/dataset", "val")