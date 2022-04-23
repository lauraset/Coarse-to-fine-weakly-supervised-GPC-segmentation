import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average = True):
    Iand1 = torch.sum(target * pred, dim=[1,2,3])
    Ior1 = torch.sum(target, dim=[1,2,3]) + torch.sum(pred, dim=[1,2,3]) - Iand1
    IoU = 1- (Iand1 / (Ior1 + 1e-8))

    if size_average==True:
        IoU = IoU.mean()
    return IoU

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
