import torch
import torch.nn as nn
import torch.nn.functional as F
from losses_pytorch.ssim_loss import SSIM
from losses_pytorch.iou_loss import IOU

class CE_MSE(nn.Module):
    def __init__(self, weight=None, beta=0.7):
        super().__init__()
        self.weight = weight
        self.beta = beta # the balance term
    def forward(self, pmask, rmask, pbd,  rbd):
        ce = F.cross_entropy(pmask, rmask, weight=self.weight)
        mse = F.mse_loss(pbd, rbd.float()/255.) # normed to 0-1
        loss = ce + self.beta*mse
        return loss


class BCE_SSIM_IOU(nn.Module):
    def __init__(self, size_average=True, issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean' if size_average else 'none')
        self.ssim = SSIM(window_size=11, size_average=size_average)
        self.iou = IOU(size_average=size_average)
        self.size_average = size_average
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        if not self.size_average:
            loss_ce = torch.mean(loss_ce, dim=[1,2,3]) # N
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim + loss_iou
        return loss


# ce and iou, 2021.11.26
class BCE_IOU(nn.Module):
    def __init__(self, size_average=True, issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean' if size_average else 'none')
        # self.ssim = SSIM(window_size=11, size_average=size_average)
        self.iou = IOU(size_average=size_average)
        self.size_average = size_average
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        if not self.size_average:
            loss_ce = torch.mean(loss_ce, dim=[1,2,3]) # N
        # loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_iou
        return loss


# ce and ssim, 2021.11.26
class BCE_SSIM(nn.Module):
    def __init__(self, size_average=True, issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean' if size_average else 'none')
        self.ssim = SSIM(window_size=11, size_average=size_average)
        # self.iou = IOU(size_average=size_average)
        self.size_average = size_average
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        if not self.size_average:
            loss_ce = torch.mean(loss_ce, dim=[1,2,3]) # N
        loss_ssim = 1-self.ssim(pmask, rmask)
        # loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim
        return loss



class BCE_SSIM_IOU_BD(nn.Module):
    def __init__(self,issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.bd = torch.nn.MSELoss()
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask, pbd, rbd):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss_bd = self.bd(pbd, rbd.float()/255.)
        loss = loss_ce + loss_ssim + loss_iou + loss_bd
        return loss


# weight by mini-batch
class bcelossweight(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ypred, ytrue):
        # ypred and ytrue are the same shape, N 1 H W
        ypred = torch.sigmoid(ypred)
        weight = torch.mean(ytrue, dim=[1,2,3]) #
        loss_pos =  ytrue*torch.log(ypred)
        loss_pos = (1-weight)*torch.mean(loss_pos, dim=[1,2,3])
        loss_neg = (1-ytrue)*torch.log(1-ypred)
        loss_neg = weight*torch.mean(loss_neg, dim=[1,2,3])
        loss = -(loss_pos+loss_neg).mean()
        return loss

if __name__=="__main__":
    # loss = BCE_SSIM_IOU(size_average=True)
    # pmask = torch.rand((5,2,32,32))
    # rmask = torch.randint(0,1,(5,2,32,32))
    # v = loss(pmask, rmask.float())
    # print(v.shape)
    # print(v)

    # loss = GCE_SSIM_IOU()
    # pmask = torch.rand((5,1,32,32)).cuda()
    # rmask = torch.randint(0,1,(5,1,32,32)).cuda()
    # v = loss(pmask, rmask)
    # print(v.shape)
    # print(v)

    loss = bcelossweight()
    pmask = torch.rand((5,1,32,32)).cuda()
    rmask = torch.randint(0,1,(5,1,32,32)).cuda()
    v = loss(pmask, rmask.float())
    print(v.shape)
    print(v)