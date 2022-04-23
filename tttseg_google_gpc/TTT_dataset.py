'''
load tuitiantu dataset filename
'''
from glob import glob
import numpy as np
import os
from os.path import join
import pandas as pd
import random

# read file and labels
labeldict = {"gpc":0, "nongpc":1}

def get_img_lab_test_lvwang(datapath, listname):
    idir = 'gpc'
    datalist = join(datapath, listname)
    if os.path.exists(datalist):
        print('datalist exists')
        return
    else:
        imgpath = join(datapath, idir, "img") # img
        assert os.path.isdir(imgpath) == True
        imglist = glob(join(imgpath, "*.png"))
        lab = labeldict[idir]
        with open(datalist,"w") as f: # append
            for img in imglist:
                mask = img.replace("img", "lab")
                f.write(img+","+ mask+","+str(lab)+"\n")

if __name__=="__main__":
    # 2021.10.21
    datapath = r'.\testvalid'
    get_img_lab_test_lvwang(datapath, "test_list_gpc.txt")





