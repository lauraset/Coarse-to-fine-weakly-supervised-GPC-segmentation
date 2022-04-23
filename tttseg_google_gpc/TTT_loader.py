'''
2021.9.6 tuitiantu
'''
import torch.utils.data as data
import albumentations as A
import torch
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join
import os
from IRN.imutils import pil_rescale
from IRN.indexing import PathIndex

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
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
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

# for seg
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
        mask_path = self.datalist.iloc[index, 1]
        mask = np.array(Image.open(mask_path))
        #lab = self.datalist.iloc[index, 2]
        # mask = mask # *(lab+1) # mask [0,1] * [1,2] => mask [0, 1, 2]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]
            # img = A.center_crop(img, crop_width=imgsize, crop_height=imgsize)
            # mask = A.center_crop(mask, crop_width=imgsize, crop_height=imgsize)
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        # mask
        mask = torch.tensor(mask).long()
        return img, mask

    def __len__(self):
        return len(self.datalist)


# for irnet, random walk
class myImageFloderclsRW(data.Dataset):
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
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.stack([img, np.flip(img, -1)], axis=0) # 2 C H W

        img = torch.from_numpy(img).float() #
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        out = {"name": img_path, "img": img, "size": (img.shape[2], img.shape[3]),
               "label": lab}
        return out

    def __len__(self):
        return len(self.datalist)


# 2021.11.1
# for seg learning with negative samples
class myImageFloder_segcls(data.Dataset):
    def __init__(self, segroot, datalist, channels=3, aug=False, num_sample = 0, returnpath=False):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.segroot = segroot # for segmentation
        self.returnpath = returnpath

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
        if self.returnpath:
            return img, mask, img_path
        else:
            return img, lab, mask

    def __len__(self):
        return len(self.datalist)


class myImageFloder_segcls_update(data.Dataset):
    def __init__(self, segroot, datalist, updatepath, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.segroot = segroot # for segmentation
        self.updatepath = updatepath

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        if "negative" in img_path:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            update = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        else:
            mask = Image.open(os.path.join(self.segroot, ibase+'_obj.png'))
            mask = np.asarray(mask)
            update = Image.open(os.path.join(self.updatepath, ibase + '_up.png'))
            update = np.asarray(update)  # updated img
            mask = A.center_crop(mask, crop_height=update.shape[0], crop_width=update.shape[1])

        ref = np.stack((mask, update), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=ref)
            img = transformed["image"]
            ref = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=ref)
            img = transformed["image"]
            ref = transformed["mask"]

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        mask = torch.from_numpy(ref[:, :, 0]).long()
        update = torch.from_numpy(ref[:, :, 1]).long()
        return img, mask, update

    def __len__(self):
        return len(self.datalist)


# consider crf into label update
class myImageFloder_segcls_update_crf(data.Dataset):
    def __init__(self, segroot, datalist, updatepath, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.segroot = segroot # for segmentation
        self.updatepath = updatepath

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB')#
        img = np.asarray(img)

        if "negative" in img_path:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            update = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        else:
            mask = Image.open(os.path.join(self.segroot, ibase+'_obj.png'))
            mask = np.asarray(mask)
            update = Image.open(os.path.join(self.updatepath, ibase + '_up.png'))
            update = np.asarray(update)  # updated img
            mask = A.center_crop(mask, crop_height=update.shape[0], crop_width=update.shape[1])

        ref = np.stack((mask, update), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=ref)
            img = transformed["image"]
            ref = transformed["mask"]

        ori_img =  img.copy()
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        mask = torch.from_numpy(ref[:, :, 0]).long()
        update = torch.from_numpy(ref[:, :, 1]).long()
        return img, mask, update, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# train updated labels from scratch
class myImageFloder_segcls_update_scratch(data.Dataset):
    def __init__(self, updatepath, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.updatepath = updatepath

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        if "negative" in img_path:
            update = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        else:
            update = Image.open(os.path.join(self.updatepath, ibase + '_up.png'))
            update = np.asarray(update)  # updated img

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=update)
            img = transformed["image"]
            update = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=update)
            img = transformed["image"]
            update = transformed["mask"]

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        update = torch.from_numpy(update).long()

        return img, update

    def __len__(self):
        return len(self.datalist)


# 2021,11.06 for lvwang detection
class myImageFloder_SEC(data.Dataset):
    def __init__(self,  camroot, datalist, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.camroot = camroot

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.camroot, ibase+'_mask.png') # mask
        mask = np.array(Image.open(mask_path)) # mask:0 (ignore), 1 (bg), 2 (fg)
        lab = self.datalist.iloc[index, 1] # for classification

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        lab = torch.tensor(lab).long()
        lab_onehot = torch.zeros((self.classes))
        lab_onehot.scatter_(0, 1-lab, 1) # lab: 0-lvwang, 1-negative
        lab_onehot[0] = 1 # the first dim; bg
        lab_onehot = lab_onehot.unsqueeze(0).unsqueeze(1)
        # mask to one-hot
        mask = torch.tensor(mask).long()
        mask_onehot = torch.zeros((self.classes+1, mask.shape[0], mask.shape[1])).long()
        mask_onehot.scatter_(0, mask.unsqueeze(0), 1)
        mask_onehot = mask_onehot[1:, :, :] # delete the first dim
        return img, mask_onehot, lab_onehot

    def __len__(self):
        return len(self.datalist)


# 2021,11.08 for jstar, rloss paper
class myImageFloder_rloss_irn(data.Dataset):
    def __init__(self,  camroot, datalist, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.camroot = camroot

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.camroot, ibase+'_irn.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8')
        # mask: 0-bg, 1-fg, 255-ignore
        mask[mask==255] = 2 # ignore

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()
        return img, mask

    def __len__(self):
        return len(self.datalist)


# 2021.11.07 IRNet
class myImageFloder_IRN(data.Dataset):
    def __init__(self, camroot, datalist, channels=3, aug=False, num_sample = 0,
                 suffix ="_irn.png", path_index = None, label_scale = 0.25):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.camroot = camroot
        self.suffix = suffix
        self.label_scale = label_scale
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        if "negative" in img_path:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        else:
            mask = Image.open(os.path.join(self.camroot, ibase + self.suffix))
            mask = np.asarray(mask)  # updated img

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        # img
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # mask
        reduced_label = pil_rescale(mask, scale=self.label_scale, order=0) # nearest
        out = {}
        out['img'] = img
        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = \
            self.extract_aff_lab_func(reduced_label)
        return out

    def __len__(self):
        return len(self.datalist)


class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


# 2021.11.08 add pseudo labels
class myImageFloder_IRN_pseudo(data.Dataset):
    def __init__(self,  camroot, datalist, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.camroot = camroot

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.camroot, ibase+'.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8') # 0,1

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        return img, mask

    def __len__(self):
        return len(self.datalist)


# 2021,11.10 RRM method, using irnet labels
# refer to jstar (building detection), the two step method
class myImageFloder_RRM_irn(data.Dataset):
    def __init__(self,  camroot, datalist, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes
        # add
        self.camroot = camroot

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        # mask: 0-bg, 1-fg, 255-ignore
        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.camroot, ibase+'_irn.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8')
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask= mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        ori_img =  img.copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # mask
        mask = torch.from_numpy(mask)

        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


class myImageFloder_path_SEC(data.Dataset):
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
        mask_path = self.datalist.iloc[index, 1]
        mask = np.array(Image.open(mask_path))
        lab = self.datalist.iloc[index, 2]
        mask[mask==2] = lab+2 # foreground. 2 or 3
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        # mask
        mask = torch.tensor(mask).long()
        return img, mask, img_path

    def __len__(self):
        return len(self.datalist)


# predict return img path
class myImageFloder_path(data.Dataset):
    def __init__(self,  datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        #self.camroot = camroot

    def __getitem__(self, index):
        img_path = os.path.join(self.datalist.iloc[index, 0])
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        mask_path = os.path.join(self.datalist.iloc[index, 1])
        mask = np.array(Image.open(mask_path))
        lab = self.datalist.iloc[index, 2]
        mask = mask*(lab+1) # mask [0,1] * [1,2] => mask [0, 1, 2]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]
        img = (img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        # mask
        mask = torch.tensor(mask).long()
        return img, mask, img_path

    def __len__(self):
        return len(self.datalist)


if __name__ =="__main__":
    a = np.ones((256,256), dtype=np.float)
    b = A.rotate(a, 15,)
    print(b.min())