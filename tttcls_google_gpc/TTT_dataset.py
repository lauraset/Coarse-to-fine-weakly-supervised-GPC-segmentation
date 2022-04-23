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

def get_file(datapath, suffix="*.tif"):
    dirlist=os.listdir(datapath)
    dirlist.sort()
    # data_list.txt
    datalist = join(datapath,'datalist.txt')
    if os.path.exists(datalist):
        print('datalist exists')
        return
    else:
        for idir in dirlist:
            imgpath = join(datapath, idir, "img") # img
            if not os.path.isdir(imgpath):
                continue
            imglist = glob(join(imgpath, suffix))
            lab = labeldict[idir]
            with open(datalist,"a") as f: # append
                for img in imglist:
                    f.write(img+","+ str(lab)+"\n")

def split_df(df, value=0, split_rate=0.9):
    df1 = df.loc[df[1]==value].sample(frac=1) # extract and shuffle
    num_train = int(len(df1)*split_rate)
    return df1[:num_train], df1[num_train:]

def split_data(datalist_path, split_rate=0.9, id='2'):
    data_dir = os.path.dirname(datalist_path)
    train_path = join(data_dir, 'train_list'+id+'.txt')
    test_path = join(data_dir, 'test_list'+id+'.txt')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('train and test list exist')
        return
    else:
        df = pd.read_csv(datalist_path, sep=',', header=None)
        # stratified random sampling
        df0_trn, df0_tst = split_df(df, 0, split_rate)
        df1_trn, df1_tst = split_df(df, 1, split_rate)
        df_train = pd.concat([df0_trn, df1_trn]).sample(frac=1) # concate and shuffle
        df_test = pd.concat([df0_tst, df1_tst]).sample(frac=1)
        df_train.to_csv(train_path, index=False, sep=',', header=None)
        df_test.to_csv(test_path, index=False, sep=',', header=None)
        print('success')

def extract_neg(list1,respath):
    df1 = pd.read_csv(list1, header=None, delimiter=',')
    df2 = df1[df1[1]==1]
    df2.to_csv(respath, header=None, index=False)

def extract_pos(list1,respath):
    df1 = pd.read_csv(list1, header=None, delimiter=',')
    df2 = df1[df1[1]==0]
    df2.to_csv(respath, header=None, index=False)

if __name__=="__main__":
    iroot = r'.\data' # contain gpc and nongpc directories
    get_file(iroot, suffix="*.png")
    datalist_path =join(iroot, 'datalist.txt')
    df = pd.read_csv(datalist_path, sep=',', header=None)
    # 60% for training, 40% for testing
    split_data(datalist_path, split_rate=0.6, id='_0.6')

    # 2021.10.22 balance training
    # extract positive and negative samples
    list1 = join(iroot, 'train_list_0.6.txt')
    extract_pos(list1, join(iroot,'train_list_0.6_gpc_pos.txt')) # lvwang
    extract_neg(list1, join(iroot,'train_list_0.6_gpc_neg.txt')) # negative

    list1 = join(iroot,'test_list_0.6.txt')
    extract_pos(list1, join(iroot,'test_list_0.6_gpc_pos.txt')) # lvwang
    extract_neg(list1, join(iroot,'test_list_0.6_gpc_neg.txt')) # negative








