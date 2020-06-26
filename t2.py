#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger
# from model import BiSeNet
from n2p import mynet
# from network import mynet
from cityscapes2 import CityScapes
from loss import OhemCELoss
# from evaluate import evaluate
from optimizer import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse
from collections import OrderedDict


logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = 0,
            )
    parse.add_argument('-c', type=str,
                       metavar="FILE",
                       dest="ckpt_path",
                       help='continue from one certain checkpoint')
    return parse.parse_args()


def train(res_savepath=None):
    print("------------train")
    if not osp.exists(res_savepath): os.makedirs(res_savepath)
    args = parse_args()
    # print(args.local_rank)
    # print("device: ",torch.cuda.device_count())
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:33666',
                world_size = torch.cuda.device_count(),
                rank=args.local_rank
                )
    setup_logger(res_savepath,0)

    ## dataset
    n_classes = 19
    n_img_per_gpu = 4
    n_workers = 16
    cropsize = [1024, 1024]
    # ds = CityScapes('/home/lyf/workdir/disk/dataset/cityscapes', cropsize=cropsize, mode='train')
    ds = CityScapes('/home/lyf/git-repo/real-time-segmentation/datasets/cityscapes/cityscapes', cropsize=cropsize, mode='train')
    
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    ## model
    ignore_idx = 255
    net = mynet(out_planes=n_classes)
    print(net)
    net.cuda()
    net.train()
    print(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank,],
            output_device = args.local_rank
            )
    score_thres = 0.7
    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr =1e-5
    optim = Optimizer(
            model = net.module,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    save_pth = osp.join(res_savepath, 'model_')
    msg_iter = 100
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    st_it = 0
    # if args.ckpt_path is not None:
    #     tmp = torch.load(args.ckpt_path,map_location=lambda storage, loc: storage.cuda(args.local_rank))
    #     state_dict = tmp['model']
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k
    #         new_state_dict[name] = v
    #     state_dict = new_state_dict
    #     net.load_state_dict(state_dict)

    #     optim.optim.load_state_dict(tmp['optimizer'])
    #     optim.it = tmp['iteration']
    #     epoch = tmp['epoch']
    #     st_it = tmp['iteration']

    for it in range(st_it, max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)
        net.train()

        optim.zero_grad()
        out,out16, out32 = net(im)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss3 = criteria_32(out32, lb)
        loss = 2*lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        ## print training log message
        if (it+1)%msg_iter==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if  it>78000:
            if (it)%200==0 or ((it)%50==0 and it>79000):
                state_dict = {}
                state_dict['model'] = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                state_dict['optimizer'] = optim.optim.state_dict()
                state_dict['iteration'] = it
                state_dict['epoch'] = epoch
                net_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank()==0: 
                    name = save_pth+str(it)+'.pth'
                    torch.save(state_dict, name)
                    # if it >70000:
                    #     n2=save_pth+str(it)+'_2.pth'
                    #     torch.save(state_dict['model'], n2)
                    # torch.save(state_dict, save_pth+'final.pth')
                    logger.info('training done, model saved to: {}'.format(name))
                #if it>=36000:
                 #   evaluate(respth=res_savepath ,e=it)

    ## dump the final model
    save_pth = osp.join(res_savepath, 'new.pth')
    #net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank()==0: torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    path ='./res18-loss'
    train(path)
    # evaluate(respth=path ,e='final')
#path, tcp,
