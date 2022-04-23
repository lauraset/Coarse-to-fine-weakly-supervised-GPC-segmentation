# 2021.10.18: upsupervised segmentation
# method: infoseg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" # choose GPU:0

import time

import cv2
import numpy as np
from skimage import segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
from tqdm import tqdm
from torch.autograd import Variable

class Args(object):
    scribble = False
    nChannel = 100
    maxIter = 2**6
    minLabels = 3
    lr = 0.1
    nConv = 2
    stepsize_sim = 1  #'step size for similarity loss'
    stepsize_con = 1  # 'step size for continuity loss'
    stepsize_scr = 0.5 # 'step size for scribble loss'


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim, nChannel, nConv):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)
        self.nConv = nConv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


def run(input_image_path, resdir="unseg"):
    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    ibase = os.path.basename(input_image_path)[:-4] # remove the suffix
    idir = os.path.dirname(os.path.dirname(input_image_path))
    respath = os.path.join(idir, resdir, ibase)
    if os.path.exists(respath+'.png'):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # load image
    im = cv2.imread(input_image_path)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) ).to(device)
    data = Variable(data)

    # train
    model = MyNet(input_dim=data.size(1), nChannel=args.nChannel, nConv=args.nConv).to(device)
    model.train()
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()
    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()
    # continuity loss definition
    loss_hpy = torch.nn.L1Loss()
    loss_hpz = torch.nn.L1Loss()

    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel).to(device)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # t1=time.time()
    for batch_idx in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

        outputHP = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)

        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # loss
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
        loss.backward()
        optimizer.step()

        if nLabels <= args.minLabels:
            print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break
    # print('%.3f s'%(time.time()-t1))

    # save output image
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    target = torch.argmax( output, 1 )
    im_target = target.data.cpu().numpy()
    un_label, lab_inverse = np.unique(im_target, return_inverse=True, )

    img_flatten = im.reshape((-1, 3)).copy()
    color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
    for lab_id, color in enumerate(color_avg):
        img_flatten[lab_inverse == lab_id] = color
    show = img_flatten.reshape(im.shape)
    lab_inverse = lab_inverse.reshape((im.shape[0], im.shape[1]))
    cv2.imwrite(respath+'_c.png', show)
    cv2.imwrite(respath+'.png', lab_inverse)


if __name__ == '__main__':

    df = pd.read_csv(r'.\data\test_list_0.6_gpc_pos.txt', header=None,sep=',')
    filelist = df[0].tolist()
    os.makedirs(r'.\data\gpc\unseg_v1', exist_ok=True)
    for input_image_path in tqdm(filelist):
        run(input_image_path, resdir='unseg_v1')

    # multi-processing
    # t1 = time.time()
    # pool = Pool(8)
    # pool.map(run, filelist)
    # pool.close()
    # pool.join()
    # print('elaps: %.3f'%(time.time()-t1))
