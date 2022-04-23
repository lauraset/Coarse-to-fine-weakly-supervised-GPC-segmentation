import numpy as np
from scipy.ndimage import zoom
from .util_dsrg import CC_lab
import torch
import warnings
from lib.crf import crf_inference

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
MIN_PROB = 1e-4
##############################################################################
  
def softmax_layer(preds):
    preds = preds
    pred_max, _ = torch.max(preds, dim=1, keepdim=True)
    pred_exp = torch.exp(preds - pred_max.clone().detach())
    probs = pred_exp / torch.sum(pred_exp, dim=1, keepdim=True) + MIN_PROB
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs # bx21x41x41

##############################################################################

def single_generate_seed_step(params):
    
    #th_f,th_b = 0.85,0.99
    data, threshold_config = params
    target, label, prob = data
    th_f, th_b = threshold_config
    
    existing_prob = prob*target
    existing_prob_argmax = torch.argmax( existing_prob, dim=2) + 1 # to tell the background pixel and the not-satisfy-condition pixel
    tell_where_is_foreground_mask = (existing_prob_argmax > 1).type(torch.uint8)

    existing_prob_fg_th_mask = (torch.sum((existing_prob[:,:,1:] > th_f).type(torch.uint8), dim=2) > 0.5).type(torch.uint8)
    existing_prob_bg_th_mask = (torch.sum((existing_prob[:,:,0:1] > th_b).type(torch.uint8), dim=2) > 0.5).type(torch.uint8)
    
    map1 = existing_prob_fg_th_mask*tell_where_is_foreground_mask
    map2 = existing_prob_bg_th_mask*(1-tell_where_is_foreground_mask)
    label_map = (map1 + map2)*existing_prob_argmax
    
    label_map = label_map.cpu().numpy()
    target = target.cpu().numpy()
    label = label.cpu().numpy()
    cls_index = np.where(target > 0.5)[2] 
    for c in cls_index:
        mat = (label_map == (c+1))
        mat = mat.astype(int)
        cclab = CC_lab(mat)
        cclab.connectedComponentLabel() 
        high_confidence_set_label = set() 
        for (x,y), value in np.ndenumerate(mat):
            if value == 1 and label[x,y,c] == 1:
                high_confidence_set_label.add(cclab.labels[x][y])
            elif value == 1 and np.sum(label[x,y,:]) == 1:
                cclab.labels[x][y] = -1
        for (x,y),value in np.ndenumerate(np.array(cclab.labels)):
            if value in high_confidence_set_label:
                label[x,y,c] = 1
                
    label = torch.from_numpy(label)
    return torch.unsqueeze(label, 0)
    
def dsrg_layer(targets, labels, probs_ori, num_classes, thre_fg, thre_bg, pool):

    targets = torch.reshape(targets,(-1,1,1,num_classes))
    labels = torch.transpose(labels, 1, 3) # bx41x41x21
    probs = torch.transpose(probs_ori, 1, 3) # bx41x41x21
    targets[:,:,:,0] = 1 
    # probs = probs.clone().detach()
    
    batch_size = targets.shape[0]
    complete_para_list = []
    for i in range(batch_size):
        params_list = []
        params_list.append([targets[i],labels[i],probs[i]])
        params_list.append([thre_fg, thre_bg])
        complete_para_list.append(params_list)  
    
    # ret = pool.map(single_generate_seed_step,complete_para_list)
    ret = []
    for i in range(batch_size):
        ret.append(single_generate_seed_step(complete_para_list[i]))

    new_labels = ret[0]
    for i in range(1,batch_size):
        new_labels = torch.cat([new_labels,ret[i]], dim=0)
    new_labels = torch.transpose(new_labels, 1, 3) # bx21x41x41
    return new_labels    


def dsrg_seed_loss_layer(probs, labels):
    
    count_bg = torch.sum(labels[:,0:1,:,:], dim=(2, 3, 1), keepdim=True)
    loss_bg = -torch.mean(torch.sum(labels[:,0:1,:,:] * torch.log(probs[:,0:1,:,:]), dim=(2, 3, 1), keepdim=True) / (count_bg+1e-8))
    
    count_fg = torch.sum(labels[:,1:,:,:], dim=(2, 3, 1), keepdim=True)
    loss_fg = -torch.mean(torch.sum(labels[:,1:,:,:] * torch.log(probs[:,1:,:,:]), dim=(2, 3, 1), keepdim=True) / (count_fg+1e-8))
    loss_balanced = loss_bg+loss_fg
    
    count_bg_avg = torch.mean(count_bg.squeeze().float())
    count_fg_avg = torch.mean(count_fg.squeeze().float())
    return loss_balanced, loss_bg, count_bg_avg, loss_fg, count_fg_avg

##############################################################################
# def crf_layer(fc8_SEC, images, iternum):
#     unary = np.transpose(np.array(fc8_SEC.cpu().clone().data), [0, 2, 3, 1])
#     mean_pixel = np.array([104.0, 117.0, 123.0])
#     im = images.cpu().data
#     im = zoom(im, (1, 1, 41 / im.shape[2], 41 / im.shape[3]), order=1)
#
#     im = im + mean_pixel[None, :, None, None]
#     im = np.transpose(np.round(im), [0, 2, 3, 1])
#
#     N = unary.shape[0]
#
#     result = np.zeros(unary.shape)
#
#     for i in range(N):
#         result[i] = CRF(im[i], unary[i], maxiter=iternum, scale_factor=12.0)
#     result = np.transpose(result, [0, 3, 1, 2])
#     result[result < MIN_PROB] = MIN_PROB
#     result = result / np.sum(result, axis=1, keepdims=True)
#
#     return np.log(result)

crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":20, "bi_srgb":13, "bi_compat":3, "iterations":5}
IMG_MEAN_ALL = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL = np.array([52.7343, 45.8798, 44.3465])

# revised by yinxcao
def crf_layer(fc8_SEC, images, iternum):
    unary = np.asarray(fc8_SEC.cpu().clone().data)  # N C H W
    N = unary.shape[0]
    C = unary.shape[1]
    # mean_pixel = np.array([104.0, 117.0, 123.0])
    im = images.cpu().data
    # im = zoom(im, (1, 1, 41 / im.shape[2], 41 / im.shape[3]), order=1)

    # im = im + mean_pixel[None, :, None, None]
    im = np.transpose(np.round(im.numpy()), [0, 2, 3, 1])

    result = np.zeros(unary.shape)

    for i in range(N):
        tmp =  im[i] * IMG_STD_ALL + IMG_MEAN_ALL
        tmp = np.ascontiguousarray(tmp)
        result[i] = crf_inference(img=tmp, feat=unary[i], crf_config=crf_config,
                                  category_num=C)
    # result = np.transpose(result, [0, 3, 1, 2])
    result[result < MIN_PROB] = MIN_PROB
    result = result / np.sum(result, axis=1, keepdims=True)

    return np.log(result)


def constrain_loss_layer(probs, probs_smooth_log):
    probs_smooth = torch.exp(probs.new_tensor(probs_smooth_log, requires_grad=True))
    loss = torch.mean(torch.sum(probs_smooth * torch.log(probs_smooth / probs), dim=1))

    return loss
