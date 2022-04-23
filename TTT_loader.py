'''
2021.9.6 tuitiantu
'''
import torch.utils.data as data
import albumentations as A
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os

# imgsize for google
imgsize = 512
image_transform = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# for train
image_transform_randomcrop = A.Compose([
    A.RandomCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
   # A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# for test
image_transform_test = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
]
)
# # used for high resolution images 1 m
# IMG_MEAN_ALL = np.array([461.7179, 350.8342, 295.0500, 258.8854])
# IMG_STD_ALL = np.array([135.3398, 116.8479, 117.2889, 108.3638])
# # for uint8
# IMG_MEAN_ALL_low = np.array([64.7314  ,70.5687 ,  74.6582 ,  90.3355])
# IMG_STD_ALL_low = np.array([33.4498 ,  35.7775 ,  38.0263 ,  36.7164])
# for google img, RGB
IMG_MEAN_ALL = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL = np.array([52.7343, 45.8798, 44.3465])


def norm_totensor(img, mean=IMG_MEAN_ALL, std=IMG_STD_ALL, channels=3):
    img = (img - mean[:channels]) / std[:channels]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
    return img


class myImageFloder(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]
        else:
            img = image_transform_test(image=img)["image"] # centercrop
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab

    def __len__(self):
        return len(self.datalist)


# for seg and cls model
class myImageFloder_segcls(data.Dataset):
    def __init__(self, segroot, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.segroot = segroot # for segmentation

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        lab = self.datalist.iloc[index, 1] # 0 or 1

        if "negative" in img_path:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        else:
            mask = Image.open(os.path.join(self.segroot, ibase+'_obj.png'))
            mask = np.asarray(mask)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long()
        return img, lab, mask

    def __len__(self):
        return len(self.datalist)


# return imgpath
class myImageFloder_path(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]
        else:
            pass
            # img = image_transform_test(image=img)["image"] # centercrop
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab, img_path

    def __len__(self):
        return len(self.datalist)


# return img and imgpath
class myImageFloder_img_path(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]
        else:
            img = image_transform_test(image=img)["image"] # centercrop
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        return img, img_path

    def __len__(self):
        return len(self.datalist)

