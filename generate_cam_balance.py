## generate cues from grad-cam++
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from tqdm import tqdm
import torch.nn.functional as F
from gradcam import GradCAMPlusPlus
import numpy as np
from TTT_loader import myImageFloder_img_path
from model import ClassificationModel
from scipy import ndimage
import cv2
import tifffile as tif
import shutil
from PIL import Image
import pandas as pd

IMG_MEAN = np.array([98.3519, 96.9567, 95.5713])
IMG_STD = np.array([52.7343, 45.8798, 44.3465])

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = (img-IMG_MEAN)/IMG_STD
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
    return img.unsqueeze(0)


# generate grad-cam
def gen_cam_lvwang(model, dataloader, train_cues_dir, backbone='regnet',
         device='cuda', aug_smooth = False, eigen_smooth=False):
    # model: e.g., vgg-16
    # dataloader: generate batch of imgs, and their path
    # train_cues_dir: save path
    # confidence: class confidence thresholds, where the last dim is background
    os.makedirs(train_cues_dir, exist_ok=True)
    os.makedirs(os.path.join(train_cues_dir, 'gpc'), exist_ok=True)
    # os.makedirs(os.path.join(train_cues_dir, 'jianshe'), exist_ok=True)
    # labelfile = os.path.join(train_cues_dir, 'labels.txt')
    # f = open(labelfile, 'a')
    # num_class = len(confidence)
    localization_cues = {}#
    target_layer=''
    if backbone=="regnet":
        target_layer = [model.encoder.s4]
    elif backbone=="resnet":
        target_layer = [model.encoder.layer4]
    else:
        raise Exception("Error in set target_layer")

    cam_method = GradCAMPlusPlus(model=model, target_layers=target_layer,
                    use_cuda=True if device == "cuda" else False)
    # Process by batch
    for img, imgpath in tqdm(dataloader):
        # img = preprocess(imgpath)
        img = img.to(device, non_blocking=True)
        pred_scores = model(img)
        pred_scores = F.softmax(pred_scores, dim=1) # N C
        pred_scores = pred_scores.cpu().detach().numpy() # N C
        # generate grad-cam for a batch of img. H: shape (N C H W)
        # xsize = img.shape[0]
        # run a batch of imgs and for all classes
        H = cam_method(input_tensor=img, target_category=[0],
                       aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)
        # save grad_cam and pred_scores
        for i, imp in enumerate(imgpath):
            iname = os.path.basename(imp)[:-4]
            idir = os.path.basename(os.path.dirname(os.path.dirname(imp)))# "lvwang
            icue = os.path.join(train_cues_dir, idir, iname)
            j=0
            h = H[i, :, :] # N H W C
            tif.imwrite(icue+'_%d_%.3f.tif'%(j, pred_scores[i,j]), h) #
            shutil.copy(imp, icue+'.png')


def main():
    #
    train_cues_dir = r'.\pred'

    nchannels = 3
    classes = 2
    device ='cuda'
    trainlist = r'.\data\test_list_0.6_gpc_pos.txt'
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder_img_path(trainlist, aug=False, channels=nchannels),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    imgpathlist = pd.read_csv(trainlist, header=None, sep=',')
    imgpathlist = imgpathlist[0].values.tolist()
    # use balance model
    net = ClassificationModel(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                             in_channels=nchannels, classes=classes).to(device)
    pretrainp = r'.\runs\regnet040_0.6_balance\model_best.tar'
    if not os.path.exists(pretrainp):
        return
    net.load_state_dict(torch.load(pretrainp)["state_dict"])
    net.eval() # keep the batchnorm layer constant

    # target: 0==>gpc
    gen_cam_lvwang(model=net, dataloader=traindataloader, train_cues_dir=train_cues_dir,
             backbone='regnet', device=device, aug_smooth=False, eigen_smooth=False)


if __name__=="__main__":
    main()