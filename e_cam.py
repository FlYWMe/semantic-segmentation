#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
#from n2p import mynet
from n2p_cam import mynet
from camvid import Camvid
#from testnet import mynet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import argparse
import cv2
from torchscope import scope
# from compute_speed import compute_speed  

class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            dsval,
            istest,
            scales = [1],
            #scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 11,
            lb_ignore = 11,
            cropsize = 480,
            flip = False,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.dsval = dsval
        self.net = model
        self.istest = istest


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
            # print(prob.shape)
        return prob

    def crop_eval(self, im):
        # cropsize = self.cropsize
        # stride_rate = 5/6.
        # N, C, H, W = im.size()
        # long_size, short_size = (H,W) if H>W else (W,H)
        # if long_size < cropsize:
        #     im, indices = self.pad_tensor(im, (cropsize, cropsize))
        #     prob = self.eval_chip(im)
        #     prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        # else:
            # stride = math.ceil(cropsize*stride_rate)
            # if short_size < cropsize:
            #     if H < W:
            #         im, indices = self.pad_tensor(im, (cropsize, W))
            #     else:
            #         im, indices = self.pad_tensor(im, (H, cropsize))
        N, C, H, W = im.size()
            # n_x = math.ceil((W-cropsize)/stride)+1
            # n_y = math.ceil((H-cropsize)/stride)+1
        prob = torch.zeros(N, self.n_classes, H, W).cuda()
        prob.requires_grad = False
            # for iy in range(n_y):
            #     for ix in range(n_x):
            #         hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
            #         hst, wst = hed-cropsize, wed-cropsize
            #         chip = im[:, :, hst:hed, wst:wed]
        prob = self.eval_chip(im)
                    # print(prob_chip.shape)
                    # prob[:, :, hst:hed, wst:wed] += prob_chip
                    # prob += prob_chip
        # if short_size < cropsize:
        #     prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        # print(cropsize)
        # print(prob.shape)
        return prob

    def scale_crop_eval(self, im, scale):
        # cropsize = self.cropsize
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        # print(pred.shape)
        # print(lb.shape)
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self):
        ## evaluate
        cnt=0
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label,fn) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                # print(imgs)
                # print(label)
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            # newlabel,newname = self.dsval.transform_label( preds, fn[0])
            # l =label.data.numpy().squeeze(1)
            # print(l.shape)
            # if fn[0] == "berlin_000478_000019":
            # if cnt==0:
            #     fpixel=open('s/'+fn[0]+'.txt','w')
            #     # print(newlabel.shape[1],newlabel.shape[2])
            # ll =[]
            # for i in range(preds.shape[1]):
            #     for j in range(preds.shape[2]):
            #         if preds[0][i][j] >10:#not in ll:
            #     	# ll.append(preds[0][i][j])
            #             print(preds[0][i][j])
            #             fpixel.write(str(newlabel[0][i][j])+' ')
            #             cnt+=10
            # if self.istest==1:
            #     newlabel,newname = self.dsval.transform_label( preds, fn[0])
            #     print(fn[0], newname)
            #     newlabel=np.squeeze(newlabel)
            #     newpath='mfinal/'+newname
            #     cv2.imwrite(newpath, newlabel)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once

        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(respth='./res1024', dspth='/home/lyf/git-repo/BiSeNet/camvid/',e='final', istest=0):
    ## logger
    logger = logging.getLogger()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--epochs', default='last', type=str)
    # args = parser.parse_args()
    #models = os.path.join(respth,'model_final.pth')
    models = os.path.join(respth,'model_%s.pth' % e)
    logger.info('load models %s\n' %models)

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 11
    net = mynet(n_classes)
    # save_pth = osp.join(respth, 'model_70400.pth')
    state_dict = torch.load(models)
    state_dict = state_dict['model']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
        # name = k[7:]
        # new_state_dict[name] = v
    # state_dict = new_state_dict
    net.load_state_dict(state_dict)
    # net.load_state_dict(torch.load(save_pth))
     
    #scope(net, input_size=(3, 768, 1536))
    #compute_speed(net,(1,3,768,1536), 0, 1000) 
    net.cuda()
    net.eval()

    ## dataset
    if istest==1:
        batchsize=1
        n_workers = 1
        dsval = Camvid(dspth, mode='test')  #
    else:
        batchsize =6
        n_workers = 2
        dsval = Camvid(dspth, mode='val')  #
    # cropsize = [1024, 1024]
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl,dsval,istest)
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    args = parser.parse_args()

    respath = './n2p-camvid'
    # resepoch = '79900_2'
    setup_logger(respath, args.epochs)
    evaluate(respath, e=args.epochs,istest=1)
