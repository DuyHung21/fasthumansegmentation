import os
from PIL import Image
from torch.utils import data

class HumanDataset(data.Dataset):
    def __init__(self, file_list, transform=None):
        self._transform = transform

        if isinstance(file_list, str):
            with open(file_list, 'r') as f:
                self.files = f.readlines()
        else:
            self.files = file_list

        #self.files = self.files[:64]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img, mask, is_resize = self._get_img_mask(idx)
        # mask = mask.convert('P')
        matt = mask.copy()

        _img, _mask, _matt = (self._transform(img, mask, matt, is_resize))
        # print(_img.shape, _mask.shape, _matt.shape)

        return _img, _mask, _matt

    def _get_img_mask(self, idx):
        train_path, label_path = self.files[idx].replace('\n', '').split(',')

        dir_path = train_path.split('/')[8]
        #print(dir_path, train_path.split('/')[-1], label_path)

        img = Image.open(train_path.strip())
        mask = Image.open(label_path.strip())

        #temp = mask.copy()
        #temp[temp > 0] = 255
        #cv2.imshow('data_origin', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imshow('label_origin', temp.astype(np.uint8))
        #print(file_name)

        # return img, mask, dir_path == 'Supervisely_Person_Dataset'
        return img, mask, False
    
if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision import transforms

    from custom_transforms import CustomTransform, Colorize
    import matplotlib.pyplot as plt
    import numpy as np
    from os.path import dirname, abspath

    def imshow(inp, title=True):
        """Imshow for tensor"""
        inp = inp.numpy().transpose(1, 2, 0)
        mean = np.array([0.4505, 0.4137, 0.3889])
        std = np.array([0.2447, 0.2328, 0.2296])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    parrent_dir = dirname(dirname(abspath(dirname(abspath(__file__)))))
    files = os.path.join(parrent_dir, 'SegmentationData/CustomData/train_nococo.txt')

    color_transform = Colorize(n=2)
    image_transform = transforms.ToPILImage()

    dataset = HumanDataset(files, CustomTransform(crop_size=(256, 256), flip=True))
    dataloaders = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    inputs, masks, matts = next(iter(dataloaders))
    print(inputs.shape)
    #inputs = [image_transform(x) for x in inputs]
    out = torchvision.utils.make_grid(inputs)
    out1 = torchvision.utils.make_grid(masks)
    out2 = torchvision.utils.make_grid(matts)
    plt.subplot(3, 1, 1)
    imshow(out, title=False)
    plt.subplot(3, 1, 2)
    imshow(out2, title=False)
    plt.subplot(3, 1, 3)
    imshow(out2, title=False)
    print(np.unique(out2))
    plt.show()
