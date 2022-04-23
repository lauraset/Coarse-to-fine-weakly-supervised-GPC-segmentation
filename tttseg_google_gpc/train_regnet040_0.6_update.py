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
from TTT_loader import myImageFloder_segcls, myImageFloder, \
    myImageFloder_segcls_update,\
    myImageFloder_segcls_update_scratch
import segmentation_models_pytorch  as smp
from metrics import SegmentationMetric, AverageMeter
import cv2 # for update
import shutil
from myloss import BCE_SSIM_IOU

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 15:
        lr = 0.001
    elif epoch <= 30:
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
    iroot = r'..\tttcls_google_gpc\data'
    trainlist_pos = os.path.join(iroot, 'train_list_0.6_gpc_pos.txt')
    trainlist_neg = os.path.join(iroot, 'train_list_0.6_gpc_neg.txt')
    segroot = os.path.join(iroot, 'pred')
    testlablist = r'.\testvalid\test_list_gpc.txt'
    # update path for store the update labels
    updatepath = os.path.join(segroot, "gpc_update")
    os.makedirs(updatepath, exist_ok=True)

    # Setup parameters
    batch_size = 6
    numworkers = 4
    epochs_warmup = 10
    epochs_correct = 20
    epochs_scratch = 35
    classes = 2
    nchannels = 3
    device = 'cuda'
    iswarm = True
    isupdate = True
    isscratch = True
    isneg = False # not use negative samples
    global best_acc
    best_acc =0

    logdirwarm = r'.\runs\regnet040_objbase_update\warm'
    os.makedirs(logdirwarm, exist_ok=True)

    logdirup = r'.\runs\regnet040_objbase_update\update'
    os.makedirs(logdirup, exist_ok=True)

    logdirscratch = r'.\runs\regnet040_objbase_update\scratch'

    # 1. for warm up
    if iswarm:
        traindataloader_pos = torch.utils.data.DataLoader(
            myImageFloder_segcls(segroot, trainlist_pos, aug=True, channels=nchannels),
            batch_size=batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)

    # 2. for generate update labels
    if isupdate:
        traindataloader_pos_path = torch.utils.data.DataLoader(
            myImageFloder_segcls(segroot, trainlist_pos, aug=False, channels=nchannels, returnpath=True),
            batch_size=batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)
        traindataloader_pos_update = torch.utils.data.DataLoader(
            myImageFloder_segcls_update(segroot, trainlist_pos, updatepath, aug=True, channels=nchannels),
            batch_size=batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)

    # 3. for train with updated labels from scratch
    if isscratch:
        batch_size_train = batch_size
        if isneg:
            batch_size_train = batch_size//2
        traindataloader_pos_update_scratch = torch.utils.data.DataLoader(
            myImageFloder_segcls_update_scratch(updatepath, trainlist_pos , aug=True, channels=nchannels),
            batch_size=batch_size_train, shuffle=True, num_workers=numworkers, pin_memory=True)
        traindataloader_neg = torch.utils.data.DataLoader(
            myImageFloder_segcls_update_scratch(updatepath, trainlist_neg, aug=True, channels=nchannels),
            batch_size=batch_size_train, shuffle=True, num_workers=numworkers, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder( testlablist, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = smp.Unet(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                             in_channels=nchannels, classes=1).to(device)

    # print the model
    start_epoch = 0
    resume = os.path.join(logdirscratch, 'finetune_15.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")

    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # get all parameters (model parameters + task dependent log variances)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = BCE_SSIM_IOU(issigmoid=True)

    # 1. warm up
    if iswarm:
        writer = SummaryWriter(log_dir=logdirwarm)
        for epoch in range(epochs_warmup):
            epoch = start_epoch + epoch + 1 # current epochs
            lr = optimizer.param_groups[0]['lr']
            print('epoch %d, lr: %.6f'%(epoch, lr))
            train_loss, train_oa, train_iou = train_epoch(net, criterion, traindataloader_pos,
                                                          optimizer, device, epoch, classes)
            savefilename = os.path.join(logdirwarm, 'finetune.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            }, savefilename)
            val_loss, val_oa, val_iou  = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
            # save every epoch
            # write
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('train/1.loss', train_loss,epoch)
            writer.add_scalar('train/2.oa', train_oa, epoch)
            writer.add_scalar('train/3.iou_gpc',train_iou[1], epoch)
            writer.add_scalar('train/4.iou_nongpc', train_iou[0], epoch)
            writer.add_scalar('val/1.loss', val_loss, epoch)
            writer.add_scalar('val/2.oa', val_oa, epoch)
            writer.add_scalar('val/3.iou_gpc',val_iou[1], epoch)
            writer.add_scalar('val/4.iou_nongpc', val_iou[0], epoch)

    # 2.label correct after each epoch
    if isupdate:
        writerup = SummaryWriter(log_dir=logdirup)
        for epoch in range(epochs_correct):
            # update
            print('start update label==>')
            update_label(net, traindataloader_pos_path, device, updatepath)
            # train
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_oa, train_iou =\
                train_epoch_update(net, criterion, traindataloader_pos_update,
                                   optimizer, device, epoch, classes, alpha=0.2)
            # save
            savefilename = os.path.join(logdirup, 'finetune.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            }, savefilename)
            val_loss, val_oa, val_iou  = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
            # write
            writerup.add_scalar('lr', lr, epoch)
            writerup.add_scalar('train/1.loss', train_loss,epoch)
            writerup.add_scalar('train/2.oa', train_oa, epoch)
            writerup.add_scalar('train/3.iou_gpc',train_iou[1], epoch)
            writerup.add_scalar('train/4.iou_nongpc', train_iou[0], epoch)
            writerup.add_scalar('val/1.loss', val_loss, epoch)
            writerup.add_scalar('val/2.oa', val_oa, epoch)
            writerup.add_scalar('val/3.iou_gpc',val_iou[1], epoch)
            writerup.add_scalar('val/4.iou_nongpc', val_iou[0], epoch)
        writerup.close()

    # 3. train from scratch with updated labels, and with negative samples
    if isscratch:
        writerscra = SummaryWriter(log_dir=logdirscratch)
        for epoch in range(epochs_scratch-start_epoch):
            epoch = start_epoch + epoch + 1 # current epochs
            adjust_learning_rate(optimizer, epoch)
            lr = optimizer.param_groups[0]['lr']
            print('epoch %d, lr: %.6f'%(epoch, lr))
            if isneg:
                train_loss, train_oa, train_iou = train_epoch_update_scratch_neg(net, criterion,
                                                            traindataloader_pos_update_scratch,
                                                            traindataloader_neg,
                                                              optimizer, device, epoch, classes)
            else:
                train_loss, train_oa, train_iou = train_epoch_update_scratch(net, criterion,
                                                            traindataloader_pos_update_scratch,
                                                              optimizer, device, epoch, classes)
            val_loss, val_oa, val_iou  = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
            # save every epoch
            savefilename = os.path.join(logdirscratch, 'checkpoint'+str(epoch)+'.tar')
            is_best = val_oa > best_acc
            best_acc = max(val_oa, best_acc)  # update
            torch.save({
                'epoch': epoch,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
                'val_oa': val_oa,
                'best_acc': best_acc,
            }, savefilename)
            if is_best:
                shutil.copy(savefilename, os.path.join(logdirscratch, 'model_best.tar'))
            # write
            writerscra.add_scalar('lr', lr, epoch)
            writerscra.add_scalar('train/1.loss', train_loss,epoch)
            writerscra.add_scalar('train/2.oa', train_oa, epoch)
            writerscra.add_scalar('train/3.iou_gpc',train_iou[1], epoch)
            writerscra.add_scalar('train/4.iou_nongpc', train_iou[0], epoch)
            writerscra.add_scalar('val/1.loss', val_loss, epoch)
            writerscra.add_scalar('val/2.oa', val_oa, epoch)
            writerscra.add_scalar('val/3.iou_gpc',val_iou[1], epoch)
            writerscra.add_scalar('val/4.iou_nongpc', val_iou[0], epoch)
        writerscra.close()


def update_label(net, dataloader, device, updatepath):
    net.eval()
    with torch.no_grad():
        for images, mask, imgpath in tqdm(dataloader):
            images = images.to(device, non_blocking=True) # N C H W
            mask = mask # N H W
            output = net(images)
            output_pred = (torch.sigmoid(output)>0.5).long() # Prediction
            # save
            for idx, imgp in enumerate(imgpath):
                ibase = os.path.basename(imgp)[:-4]
                resname = os.path.join(updatepath, ibase)
                tmp = output_pred[idx].squeeze().cpu().numpy().astype('uint8')# H W, [0,1]
                diff = mask[idx].numpy().astype('float')-tmp.astype('float')
                diff[diff==1] = 255 # positive
                diff[diff==-1] = 128 # negative
                cv2.imwrite(resname+'_up.png', tmp)
                cv2.imwrite(resname+'_upc.png', tmp*255)
                cv2.imwrite(resname + '_diff.png', diff.astype('uint8'))


def train_epoch_update(net, criterion, dataloader, optimizer, device, epoch, classes, alpha=0.5):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, mask, update) in enumerate(dataloader):
        images = images.to(device, non_blocking=True) # N C H W
        mask = mask.to(device, non_blocking=True).unsqueeze(1) # N 1 H W
        update = update.to(device, non_blocking=True).unsqueeze(1)

        output = net(images)

        loss = alpha*criterion(output, mask.float()) + criterion(output, update.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (torch.sigmoid(output)>0.5) # N C H W
        acc_total.addBatch(output, mask)
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


# train updated labels from scratch
def train_epoch_update_scratch(net, criterion, dataloader, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, update) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        update = update.to(device, non_blocking=True).unsqueeze(1)

        output = net(images)

        loss = criterion(output, update.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (torch.sigmoid(output)>0.5) # N C H W
        acc_total.addBatch(output, update)
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


# train updated labels from scratch with negative samples
def train_epoch_update_scratch_neg(net, criterion, dataloader_pos, dataloader_neg, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader_pos)
    pbar = tqdm(range(num), disable=False)
    neg_train_iter = iter(dataloader_neg) # negative samples

    for idx, (x1, z1) in enumerate(dataloader_pos):
        try:
            x2, z2 = neg_train_iter.next()
        except:
            neg_train_iter = iter(dataloader_neg)
            x2, z2 = neg_train_iter.next()
        # combine pos and neg
        images = torch.cat((x1, x2), dim=0).to(device, non_blocking=True) # N C H W
        update = torch.cat((z1, z2), dim=0).to(device, non_blocking=True).unsqueeze(1)

        output = net(images)

        loss = criterion(output, update.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (torch.sigmoid(output)>0.5) # N C H W
        acc_total.addBatch(output, update)
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


def train_epoch(net, criterion, dataloader, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, _, mask) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).unsqueeze(1)

        output = net(images)

        loss = criterion(output, mask.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (torch.sigmoid(output)>0.5) # N C H W
        acc_total.addBatch(output, mask)
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True).unsqueeze(1) # n c h w
            ypred = model.forward(x)

            loss = criterion(ypred, y_true.float())

            # ypred = ypred.argmax(axis=1)
            ypred = (torch.sigmoid(ypred)>0.5)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            iou = acc_total.IntersectionOverUnion()
            miou = acc_total.meanIntersectionOverUnion()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU{miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
            pbar.update()
        pbar.close()

    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


if __name__ == "__main__":
    main()