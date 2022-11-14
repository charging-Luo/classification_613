import os
import torch
from torch.utils import data
import torchvision
from albumentations import *
import albumentations.pytorch
from PIL import Image
import cv2
import numpy as np
from copy import deepcopy
from random import randint


class OCT2017(data.Dataset):
    def __init__(self, data_root_path='/home/lym/dataset/训练', augmentation=False, is_transforms=True,
                 split="train", resize_shape=(224, 224)):  # ResNet原输入大小(224, 224)  (256, 512)  (512, 512)
        self.data_root_path = data_root_path
        self.num_class = 2
        self.split = split
        self.classes = {"NORMAL": 0,  "TRA":1, }
        self.files = []
        for cls in self.classes.keys():
            self.files += os.listdir(os.path.join(self.data_root_path, self.split, cls))
        self.files = self.shuffle(self.files)  # 打乱顺序
        self.augmentation = augmentation
        self.is_transforms = is_transforms
        self.resize = torchvision.transforms.Resize(resize_shape)  # 暂时设置的与预训练值相同
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),  # Too slow
             torchvision.transforms.Normalize([0.5], [0.5])
             ])
        self.augmentations = albumentations.Compose([
            Resize(height=512, width=512, p=1),  # 先resize以免，有图片短边较小
            RandomSizedCrop(min_max_height=(400, 490), height=512, width=512, p=0.5),
            ShiftScaleRotate(p=0.8, rotate_limit=35),
            HorizontalFlip(p=0.5),
            # GridDistortion(p=0.5),
            # OneOf([
            #     IAAAdditiveGaussianNoise(p=0.5),
            #     GaussNoise(p=0.5),
            # ], p=0.2),
            # OneOf([
            #     MotionBlur(p=0.2),
            #     MedianBlur(blur_limit=3, p=0.1),
            #     Blur(blur_limit=3, p=0.1),
            # ], p=0.2),
            CLAHE(p=0.6),  # Apply Contrast Limited Adaptive Histogram Equalization to the input image
            RandomBrightnessContrast(p=0.5),  # Randomly change brightness and contrast of the input image
            RandomGamma(p=0.5),
        ])

    def __len__(self):
        count = len(self.files)
        return count

    def __getitem__(self, index):
        filestr = self.files[index]
        cls = filestr.split('-')[0]
        img_path = os.path.join(self.data_root_path, self.split, cls, filestr)
        img = Image.open(img_path)
        if self.augmentation:
            img = np.array(img)
            augmented = self.augmentations(image=img)
            img = Image.fromarray(augmented['image'])
        img = self.resize(img)  # 无论是否做数据增强，val & train 均需resize
        # resize单独操作主要是由于 train_apex.py 使用了其他函数覆盖速度较慢的 transforms.ToTensor()
        if self.is_transforms:
            img = self.transforms(img)
        return img, self.classes[cls]

    def shuffle(self, l):
        temp_lst = deepcopy(l)
        m = len(temp_lst)
        while m:
            m -= 1
            i = randint(0, m)
            temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
        return temp_lst


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    batch_size = 4
    dataset = OCT2017(data_root_path='/home/lym/dataset/训练', augmentation=False, split="train")
    class_names = ["NORMAL", "TRA"]
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        labels = np.array(labels)
        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        print(imgs.shape)
        f, ax = plt.subplots(batch_size)
        for j in range(batch_size):
            ax[j].imshow(imgs[j][:, :, 0])
            ax[j].set_title(class_names[labels[j]], fontsize=10)
        plt.show()

        a = input()
        if a == "q":
            break
        else:
            plt.close()