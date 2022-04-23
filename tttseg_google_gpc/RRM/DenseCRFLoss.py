import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import sys
# sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
# from dataloaders.custom_transforms import denormalizeimage
# import time
# from multiprocessing import Pool
# import multiprocessing
# from itertools import repeat
# import pickle
import argparse

# from rloss
class DenseCRFLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        
        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation=grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None
    

class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        return self.weight*DenseCRFLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


class CEandCRF(torch.nn.Module):
    def __init__(self, ignore_index=255, densecrfloss=1e-7, sigma_rgb=15.0,
                 sigma_xy = 100, rloss_scale=0.5):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
        self.crf = DenseCRFLoss(weight=densecrfloss, sigma_rgb=sigma_rgb,
                                     sigma_xy=sigma_xy, scale_factor=rloss_scale)
    def forward(self, ori_img, pred, seg_label, croppings):
        # ori_img:  B C H W, torch, float32
        # pred: B C H W, torch, cuda, from the segmentation network, the last layer
        # seg_label: B H W, torch, uint8,
        # croppings: B H W, torch, float32, where 1-roi region, 0- not roi region.
        # seg_label = seg_label.unsqueeze(1)
        pred_probs = torch.softmax(pred, dim=1) # B C H W
        ori_img = ori_img.float()
        croppings = croppings.float()

        dloss = self.crf(ori_img, pred_probs, croppings)
        dloss = dloss.cuda()

        celoss = self.ce(pred, seg_label) # N C H W, N H W

        return dloss+celoss


# for binary
class BCEandCRF(torch.nn.Module):
    def __init__(self, ignore_index=255, densecrfloss=1e-7, sigma_rgb=15.0,
                 sigma_xy = 100, rloss_scale=0.5):
        super().__init__()
        self.ce = torch.nn.BCEWithLogitsLoss()
        self.crf = DenseCRFLoss(weight=densecrfloss, sigma_rgb=sigma_rgb,
                                     sigma_xy=sigma_xy, scale_factor=rloss_scale)
    def forward(self, ori_img, pred, seg_label, croppings):
        # ori_img:  B C H W, torch, float32
        # pred: B C H W, torch, cuda, from the segmentation network, the last layer
        # seg_label: B H W, torch, uint8,
        # croppings: B H W, torch, float32, where 1-roi region, 0- not roi region.
        # seg_label = seg_label.unsqueeze(1)
        # pred_probs = torch.softmax(pred, dim=1) # B C H W
        pred_probs = torch.sigmoid(pred) # for binary
        # pred_probs = torch.cat((pred_probs, 1-pred_probs),dim=1)  # bug, pred represent fg (1)
        pred_probs = torch.cat((1-pred_probs, pred_probs), dim=1)
        ori_img = ori_img.float()
        croppings = croppings.float()

        dloss = self.crf(ori_img, pred_probs, croppings)
        dloss = dloss.cuda()

        celoss = self.ce(pred, seg_label) # N C H W, N H W

        return dloss+celoss


if __name__=="__main__":
    # rloss options
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--densecrfloss', type=float, default=2e-9,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15 ,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=100,
                        help='DenseCRF sigma_xy')
    args = parser.parse_args()
    # --densecrfloss
    # 2e-9
    # - -rloss - scale
    # 0.5
    # - -sigma - rgb
    # 15
    # - -sigma - xy
    # 100
    # success
    densecrflosslayer = DenseCRFLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy,
                                          scale_factor=args.rloss_scale)
    denormalized_image = torch.randint(0,244,(1,3,256,256))
    probs = torch.rand((1,2,256,256)).float().cuda()
    target = torch.randint(0,1,(1,256,256)).cuda()
    croppings = (target!=254).float()
    densecrfloss = densecrflosslayer(denormalized_image.float(), probs, croppings)
    print(densecrfloss.item())