'''
2021.09.06
@Yinxia Cao
@function: used for tuitiantu detection
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm

from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from TTT_loader import myImageFloder
from model import ClassificationModel
from metrics import ClassificationMetric, AverageMeter
import shutil

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 40:
        lr = 0.001
    elif epoch <= 60:
        lr = 0.0001
    else:
        lr = 0.00001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added


def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    iroot = r'.\data'
    trainlist_pos = os.path.join(iroot, 'train_list_0.6_gpc_pos.txt')
    trainlist_neg = os.path.join(iroot, 'train_list_0.6_gpc_neg.txt')
    testlist = os.path.join(iroot, 'test_list_0.6.txt')

    # Setup parameters
    batch_size = 12
    epochs = 80
    classes = 2 #
    nchannels = 3 # channels
    device = 'cuda'
    logdir = r'.\runs\regnet040_0.6_balance'
    global best_acc
    best_acc = 0
    writer = SummaryWriter(log_dir=logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # train & test dataloader
    traindataloader_pos = torch.utils.data.DataLoader(
        myImageFloder(trainlist_pos, aug=True, channels=nchannels),
        batch_size=batch_size//2, shuffle=True, num_workers=4, pin_memory=True)
    traindataloader_neg = torch.utils.data.DataLoader(
        myImageFloder(trainlist_neg, aug=True, channels=nchannels),
        batch_size=batch_size//2, shuffle=True, num_workers=4, pin_memory=True)
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder(testlist, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = ClassificationModel(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                             in_channels=nchannels, classes=classes).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    weights = torch.FloatTensor([0.5, 0.5]).to(device) # 1 1 10
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # print the model
    start_epoch = 0
    resume = os.path.join(logdir, 'checkpoint.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        # best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")
        # return

    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # get all parameters (model parameters + task dependent log variances)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))
        train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos, traindataloader_neg,
                                                     optimizer, device, epoch, classes)
        # validate every epoch
        val_loss, val_oa, val_f1 = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint.tar')
        is_best = val_oa > best_acc
        best_acc = max(val_oa, best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            'val_oa': val_oa,
            'val_f1': val_f1,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, savefilename)
        if is_best:
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss, epoch)
        writer.add_scalar('train/2.oa', train_oa, epoch)
        writer.add_scalar('train/3.f1_gpc', train_f1[0], epoch)
        writer.add_scalar('train/4.f1_nongpc', train_f1[1], epoch)
        writer.add_scalar('val/1.loss', val_loss, epoch)
        writer.add_scalar('val/2.oa', val_oa, epoch)
        writer.add_scalar('val/3.f1_gpc', val_f1[0], epoch)
        writer.add_scalar('val/4.f1_nongpc', val_f1[1], epoch)

    writer.close()


def train_epoch(net, criterion, dataloader_pos, dataloader_neg, optimizer, device, epoch, classes):
    net.train()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    # with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
    num = len(dataloader_pos)
    pbar = tqdm(range(num), disable=False)
    neg_train_iter = iter(dataloader_neg) # negative samples

    for idx, (x1, y1) in enumerate(dataloader_pos):
        try:
            x2, y2 = neg_train_iter.next()
        except:
            neg_train_iter = iter(dataloader_neg)
            x2, y2 = neg_train_iter.next()
        # combine pos and neg
        x = torch.cat((x1, x2), dim=0).to(device, non_blocking=True) # N C H W
        y_true = torch.cat((y1, y2), dim=0).to(device, non_blocking=True) # N H W

        ypred = net(x)

        loss = criterion(ypred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ypred = ypred.argmax(1)
        acc_total.addBatch(ypred, y_true)

        losses.update(loss.item(), x.size(0))
        oa = acc_total.OverallAccuracy()
        f1 = acc_total.F1score()
        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, F1: {lv:.3f}, {fei:.3f}'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa = oa,lv = f1[0], fei=f1[1]))
        pbar.update()
    pbar.close()
    return losses.avg, acc_total.OverallAccuracy(), acc_total.F1score()


def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)
            ypred = model.forward(x)
            loss = criterion(ypred, y_true)

            ypred = ypred.argmax(axis=1)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            f1 = acc_total.F1score()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, F1: {lv:.3f}, {fei:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, lv=f1[0], fei=f1[1]))
            pbar.update()
        pbar.close()
    oa = acc_total.OverallAccuracy()
    f1 = acc_total.F1score()
    return losses.avg, oa, f1


if __name__ == "__main__":
    main()