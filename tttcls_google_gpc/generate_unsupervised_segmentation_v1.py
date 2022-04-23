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
import pandas as pd
from multiprocessing import Pool # multi-process
import time
from tqdm import tqdm

class Args(object):
    #input_image_path = 'image/woof.jpg'  # image/coral.jpg image/tiger.jpg
    # input_image_path = 'image/img_bj_7_13.png'
    # input_image_path = 'image/1.png'
    train_epoch = 2 ** 6
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def run(input_image_path, resdir='unseg'):
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    image = cv2.imread(input_image_path)
    ibase = os.path.basename(input_image_path)[:-4] # remove the suffix
    idir = os.path.dirname(os.path.dirname(input_image_path))
    respath = os.path.join(idir, resdir, ibase)
    if os.path.exists(respath+'.png'):
        return
    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        # cv2.imshow("seg_pt", show)
        # cv2.waitKey(1)
        # print('Loss:', batch_idx, loss.item())
        if len(un_label) < args.min_label_num:
            break
    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite(respath+'_c.png', show)
    lab_inverse = lab_inverse.reshape((image.shape[0], image.shape[1])).astype('uint8')
    cv2.imwrite(respath+'.png', lab_inverse)


if __name__ == '__main__':

    # for test
    df = pd.read_csv(r'.\data\test_list_0.6_gpc_pos.txt', header=None, sep=',')
    filelist = df[0].tolist()
    os.makedirs(r'.\data\gpc\unseg_v2', exist_ok=True)
    num = len(filelist)
    for i in range(num):
        run(filelist[i], 'unseg_v2')

    # train
    # df = pd.read_csv(r'E:\yinxcao\tuitiantu\datagoogle\test_list_0.6_pos.txt', header=None,sep=',')
    # filelist = df[0].tolist()
    # num = len(filelist)
    # for i in range(num):
    #     run(filelist[num-i-1])


    # multi-processing
    # t1 = time.time()
    # pool = Pool(8)
    # pool.map(run, filelist)
    # pool.close()
    # pool.join()
    # print('elaps: %.3f'%(time.time()-t1))
