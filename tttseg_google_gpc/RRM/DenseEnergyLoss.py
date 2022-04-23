import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import sys
# sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter_batch
import argparse

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.cuda()

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


class CEandEnergy(torch.nn.Module):
    def __init__(self, ignore_index=255, densecrfloss=1e-7, sigma_rgb=15.0,
                 sigma_xy = 100, rloss_scale=0.5):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
        self.energy = DenseEnergyLoss(weight=densecrfloss, sigma_rgb=sigma_rgb,
                                     sigma_xy=sigma_xy, scale_factor=rloss_scale)
    def forward(self, ori_img, pred, seg_label, croppings):
        # ori_img:  B H W 3, torch, float32
        # pred: B C H W, torch, cuda, from the segmentation network, the last layer
        # seg_label: B H W, torch, uint8, where 255-ignore values,
        # croppings: B H W, torch, float32, where 1-roi region, 0- not roi region.
        seg_label = seg_label.unsqueeze(1)
        pred_probs = torch.softmax(pred, dim=1) # B C H W
        ori_img = ori_img.float()
        croppings = croppings.float()

        dloss = self.energy(ori_img, pred_probs, croppings, seg_label)
        dloss = dloss.cuda()

        celoss = self.ce(pred, seg_label.squeeze().long().cuda())

        return dloss+celoss


if __name__=="__main__":
    # 2021.11.10: success
    criterion = CEandEnergy()
    ori_img = torch.randint(0, 255,(2, 256, 256, 3)).byte()
    pred = torch.rand((2, 2, 256, 256)).float().cuda()
    seg_label = torch.randint(0, 2,(2, 256, 256)).byte() # uint8
    croppings = torch.ones((2, 256, 256)).byte()
    loss = criterion(ori_img, pred, seg_label, croppings)
    print(loss)
    '''
    # rloss options
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100,
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
    densecrflosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                        sigma_xy=args.sigma_xy,
                                        scale_factor=args.rloss_scale)
    # B 3 H W
    ori_img = np.random.randint(0,244,(2, 3, 256, 256)).astype('uint8')
    # B C H W
    seg = torch.rand((2, 2, 256, 256)).float().cuda()
    # B H W
    seg_label = np.random.randint(0, 1,(2, 256, 256)).astype('uint8')
    # H W B
    croppings = np.ones((256, 256, 2), np.bool)

    # run
    seg_label = np.expand_dims(seg_label,axis=1)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg,(w,h),mode="bilinear",align_corners=False)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2, 0, 1))
    dloss = densecrflosslayer(ori_img,pred_probs,croppings, seg_label)
    dloss = dloss.cuda()

    print(dloss.item())
    '''