import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import copy

from PIL import Image

class BinaryRelabel:
    def __init__(self, bg_label=0):
        self.bg_label = bg_label
    
    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor)
        tensor[tensor != self.bg_label] = 1
        return tensor


class ToLabel:
    def __call__(self, image):
        img = np.array(image)
        while len(img.shape) != 2:
            img = img[:, :, 0]
        return torch.from_numpy(img).long().unsqueeze(0)

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class ToRGB:
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        elif x.shape[0] > 3:
            return x[0:3, :, :]

        return x

class ToTest:
    def __call__(self, x, mean=[0.4253, 0.3833, 0.3589], std=[0.2465, 0.2332, 0.2289]):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])(x)

class CustomTransform:
    def __init__(self, resize_size=256, crop_size=None, 
                mean=None, std=None, flip=False):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip = flip
        self.mean = mean
        self.std = std

    def _random_crop(self, train, mask, matt):
        w, h = train.size
        th, tw = self.crop_size

        i, j, new_w, new_h = 0, 0, tw, th
        if w == tw and h == th:
            new_w = w
            new_h = h

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return  F.crop(train, i, j, new_h, new_w), \
                F.crop(mask, i, j, new_h, new_w), \
                F.crop(matt, i, j, new_h, new_w), \

    def _random_horizontal_flip(self, train, mask, matt):
        choice = random.random()
        if choice > 0.5:
            return F.hflip(train), F.hflip(mask), F.hflip(matt)
        else:
            return train, mask, matt

    def __call__(self, train, mask, matt):
        _train, _mask, _matt =  F.resize(train, self.resize_size), \
                                F.resize(mask, self.resize_size, Image.NEAREST), \
                                F.resize(matt, self.resize_size, Image.NEAREST)

        if self.crop_size:
            _train, _mask, _matt = self._random_crop(_train, _mask, _matt)

        if self.flip:
            _train, _mask, _matt = self._random_horizontal_flip(_train, _mask, _matt)

        if self.mean is not None and  self.std is not None:
            _train = transforms.Compose([
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(self.mean, self.std)
            ])(_train)
        else:
            _train = transforms.Compose([
                transforms.ToTensor(),
                ToRGB(),
            ])(_train)

        _mask = transforms.Compose([
            ToLabel(),
            BinaryRelabel()
        ])(_mask)

        _matt = transforms.Compose([
            ToLabel(),
            BinaryRelabel()
        ])(_matt)

        return _train, _mask, _matt.float()

